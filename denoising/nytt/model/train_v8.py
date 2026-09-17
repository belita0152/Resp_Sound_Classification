from __future__ import annotations
import os
import sys
import argparse
import random
import warnings
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(UTILS_ROOT))

from data.utils import wave_folder
from denoising.nytt.model.utils import (
    build_fold_wave_datasets,
    exclude_eval_sources,
    print_final_table,
    read_fold_ids,
    verify_fold_noise_bank,
)
from denoising.nytt.noise_bank.noise_bank_loader import NoiseBank
from denoising.utils.denoising_metric import energy_ratio_db, mix_at_snr, si_sdr
from denoising.utils.metric import (
    calculate_classification_metrics,
    classification_metrics_to_csv,
)
from denoising.nytt.loss import (TrainingLoss, BAND_LIMITED_STFT_FFTS,
                                DEFAULT_STFT_FFTS)

from denoising.nytt.model.model_v8 import (
    DUNet,
    VIEW_FREQ_KEEP,
    VIEW_NFFT,
    VIEW_REDUCED_DIM,
    parse_view_nfft,
    parse_view_pools,
    parse_view_freq_keep,
    NyTTClassifier,
)

warnings.filterwarnings("ignore")
PROJECT_ROOT = UTILS_ROOT
DATA_ROOT = PROJECT_ROOT.parent
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "denoising" / "utils" / "split_manifest_5x60_40.csv"
DEFAULT_NOISE_BANK_DIR = DATA_ROOT / "db" / "new_gt"

SR = 16000

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.denoise = not args.no_denoise
        self.noise_aug = self.denoise if args.noise_aug is None else bool(args.noise_aug)
        self.need_noise = self.denoise or self.noise_aug
        if self.denoise and not self.noise_aug:
            print("  ★ [경고] denoiser 는 켜져 있는데 잡음 주입이 꺼져 있다. "
                  "복원 손실의 입력과 정답이 같아져(x_in == x_target) "
                  "denoiser 가 항등함수를 배우는 것이 최적해가 된다.")
        train_ds, eval_ds = build_fold_wave_datasets(args)
        print(f"[data] fold={args.fold} train={len(train_ds):,} test={len(eval_ds):,}")
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True, num_workers=args.num_workers,
                                       pin_memory=self.device.type == "cuda")
        self.eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=self.device.type == "cuda")
        self.noise_bank = None
        if self.need_noise:
            if not Path(args.noise_bank).exists():
                raise FileNotFoundError(
                    f"noise bank 없음: {args.noise_bank}\n"
                    f"  python -m denoising.nytt.noise_bank.noise_bank_split --fold {args.noise_fold} "
                    "를 먼저 실행하세요."
                )
            noise_train_ids, noise_test_ids = read_fold_ids(
                args.split_manifest, args.noise_fold,
            )
            verify_fold_noise_bank(
                args.noise_bank,
                args.noise_fold,
                noise_train_ids,
                noise_test_ids,
            )
            self.noise_bank = NoiseBank(args.noise_bank, seed=args.seed)
            exclude_eval_sources(self.noise_bank, eval_ds.sample_ids)
            leaked = self.noise_bank.check_leakage(eval_ds.sample_ids)
            if leaked:
                raise RuntimeError(
                    f"noise bank에 test split {len(leaked)} ids 누수: "
                    f"{leaked[:10]}{' ...' if len(leaked) > 10 else ''}"
                )
        channels = tuple(int(c) for c in str(args.channels).split(",") if c.strip())
        dilations = tuple(int(d) for d in str(args.dilations).split(",") if d.strip())
        self.model_cfg = dict(in_channels=1, channels=channels,
                              kernel=args.kernel, stride=args.stride,
                              use_skip=not args.no_skip, predict=args.predict,
                              dilations=dilations)
        dunet = None if args.no_denoise else DUNet(**self.model_cfg)
        self.model: nn.Module = NyTTClassifier(
            dunet, num_classes=args.num_classes, base=args.cls_base,
            dropout=args.dropout,
            mel_normalize=args.mel_normalize, top_db=args.top_db,
            cls_dim=args.cls_dim,
            cls_pool=args.cls_pool,
            cls_views=tuple(v.strip() for v in str(args.cls_views).split(",")
                            if v.strip()),
            cls_view_pools=parse_view_pools(args.cls_view_pools),
            cls_view_freq_keep=parse_view_freq_keep(args.cls_view_freq_keep),
            cls_view_nfft=parse_view_nfft(
                args.cls_view_nfft,
                tuple(v.strip() for v in str(args.cls_views).split(",") if v.strip())),
            cls_view_nmels=parse_view_freq_keep(args.cls_view_nmels),
            cls_view_reduced_dim=args.cls_view_reduced_dim,
            cls_view_dim=parse_view_freq_keep(args.cls_view_dim),
        ).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.class_names = [
            "Normal", "Stridor", "Rhonchi", "Wheezing", "Crackle",
        ][:args.num_classes]
        w = train_ds.class_weights(args.num_classes).to(self.device)
        stft_ffts = (tuple(int(v) for v in str(args.stft_ffts).split(",") if v.strip())
                     if args.stft_ffts else None)
        stft_fmax = None if (args.stft_fmax_hz is None or args.stft_fmax_hz <= 0) \
            else float(args.stft_fmax_hz)

        self.train_loss = TrainingLoss(
            class_weights=w if args.use_class_weight else None,
            use_dnet=self.denoise,
            w_rec=args.w_rec, w_cls=args.w_cls,
            w_l1=args.w_l1, w_stft=args.w_stft,
            w_sisdr=args.w_sisdr,
            sr=SR,
            stft_ffts=stft_ffts, stft_fmax_hz=stft_fmax,
            aux_ce_weight=args.aux_ce_weight,
            aux_use_class_weight=args.aux_use_class_weight,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=args.epochs)
        self.amp = (self.device.type == "cuda") and not args.no_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    def _make_pair(self, x_target, snr_db=None):
        b, _, t = x_target.shape
        noise = torch.as_tensor(self.noise_bank.sample(t, batch=b),
                                dtype=torch.float32,
                                device=x_target.device).unsqueeze(1)
        if snr_db is None:
            snr_db = (torch.rand(b, device=x_target.device)
                      * (self.args.snr_max - self.args.snr_min) + self.args.snr_min)
        return mix_at_snr(x_target, noise, snr_db), noise

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total, total_rec, total_cls, n = 0.0, 0.0, 0.0, 0
        tr_preds, tr_reals = [], []
        for batch_idx, (data, target) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad(set_to_none=True)
            x_target = data.to(torch.float32).to(self.device, non_blocking=True)
            y = target.long().to(self.device, non_blocking=True)
            if self.noise_aug:
                with torch.no_grad():
                    x_in, _ = self._make_pair(x_target)
            else:
                x_in = x_target
            with torch.cuda.amp.autocast(enabled=self.amp):
                x_hat, logits = self.model(x_in)
                loss, loss_rec, loss_cls = self.train_loss(
                    x_hat, x_target, logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss: epoch={epoch}, batch={batch_idx}"
                )
            self.scaler.scale(loss).backward()
            if self.args.grad_clip and self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip,
                    error_if_nonfinite=True,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bs = x_target.size(0)
            with torch.no_grad():
                tr_preds.append(logits.detach().float().argmax(dim=1).cpu())
                tr_reals.append(y.cpu())
            total += float(loss.detach()) * bs
            total_rec += float(loss_rec.detach()) * bs
            total_cls += float(loss_cls.detach()) * bs
            n += bs
            if self.args.max_train_batches and batch_idx >= self.args.max_train_batches:
                break
        self.scheduler.step()

        n = max(n, 1)
        out = {"train_loss": total / n, "train_rec": total_rec / n,
               "train_cls": total_cls / n}
        if tr_reals:
            cls_tr = calculate_classification_metrics(
                torch.cat(tr_preds).numpy(), torch.cat(tr_reals).numpy(),
                num_classes=self.args.num_classes)
            print(f"[Train {epoch:03d}] loss={out['train_loss']:.4f} "
                  f"Acc={cls_tr['accuracy']*100:.2f} "
                  f"F1={cls_tr['f1_macro']*100:.2f} "
                  f"Se={cls_tr['sensitivity_macro']*100:.2f} "
                  f"Sp={cls_tr['specificity_macro']*100:.2f} "
                  f"ICBHI={cls_tr['score_icbhi']*100:.2f} "
                  f"(Se {cls_tr['se_icbhi']*100:.2f}"
                  f"/Sp {cls_tr['sp_icbhi']*100:.2f})")
            out.update({f"train_cls_{k}": v for k, v in cls_tr.items()
                        if np.isscalar(v)})
        return out
    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int, tag: str | None = None):
        self.model.eval()
        acc = dict(cls_loss=0.0)
        if self.denoise:
            acc.update(si_sdr_in=0.0, si_sdr_out=0.0, l1=0.0, identity_si_sdr=0.0)
        supp, retain, n_count, n_batch = 0.0, 0.0, 0, 0
        preds_all, reals_all, probs_all = [], [], []
        saved_bank_rng = None
        if self.args.fix_eval_rng:
            if self.need_noise:
                saved_bank_rng = self.noise_bank.rng
                self.noise_bank.rng = np.random.default_rng(self.args.seed)
        else:
            torch.manual_seed(self.args.seed)
            if self.need_noise:
                self.noise_bank.rng = np.random.default_rng(self.args.seed)
        dunet = self.core.dunet
        for batch_idx, (data, target) in enumerate(self.eval_loader, start=1):
            x_target = data.to(torch.float32).to(self.device, non_blocking=True)
            y = target.long().to(self.device, non_blocking=True)
            x_id, logits = self.model(x_target)
            bs = x_target.size(0)
            acc["cls_loss"] += float(self.train_loss.criterion(logits, y)) * bs
            n_count += bs
            preds_all.append(logits.argmax(dim=1).detach().cpu())
            reals_all.append(y.detach().cpu())
            probs_all.append(torch.softmax(logits.float(), dim=1).detach().cpu())
            if self.denoise:
                snr = torch.full((bs,), self.args.eval_snr, device=self.device)
                x_in, noise = self._make_pair(x_target, snr_db=snr)
                x_hat, _ = self.model(x_in)
                acc["si_sdr_in"] += si_sdr(x_in, x_target).mean().item() * bs
                acc["si_sdr_out"] += si_sdr(x_hat, x_target).mean().item() * bs
                acc["l1"] += (x_hat - x_target).abs().mean().item() * bs
                supp += energy_ratio_db(dunet, noise)
                retain += -energy_ratio_db(dunet, x_target)
                acc["identity_si_sdr"] += si_sdr(x_id, x_target).mean().item() * bs
                n_batch += 1
            if self.args.max_eval_batches and batch_idx >= self.args.max_eval_batches:
                break
        if saved_bank_rng is not None:
            self.noise_bank.rng = saved_bank_rng
        r = {k: v / max(n_count, 1) for k, v in acc.items()}
        if self.denoise:
            r["si_sdri"] = r["si_sdr_out"] - r["si_sdr_in"]
            r["noise_suppression_db"] = supp / max(n_batch, 1)
            r["signal_retention_db"] = retain / max(n_batch, 1)
        y_pred = torch.cat(preds_all).numpy()
        y_true = torch.cat(reals_all).numpy()
        y_prob = torch.cat(probs_all).numpy()
        self._last_scores = (y_prob, y_true)
        cls = calculate_classification_metrics(y_pred, y_true,
                                               num_classes=self.args.num_classes,
                                               probs=y_prob)

        r.update({f"cls_{k}": v for k, v in cls.items() if not isinstance(v, list)})
        head = (f"[Epoch]: {epoch:03d} => " if tag is None else f"[{tag}] ")
        print(head
              + f"[Acc] : {cls['accuracy']*100:.2f} "
                f"[F1] : {cls['f1_macro']*100:.2f} "
                f"[Se] : {cls['sensitivity_macro']*100:.2f} "
                f"[Sp] : {cls['specificity_macro']*100:.2f} "
                f"[ICBHI] : {cls['score_icbhi']*100:.2f} "
                f"[AUROC] : {cls.get('auroc_macro', float('nan')):.3f} "
                f"[AUPRC] : {cls.get('auprc_macro', float('nan')):.3f}")
        if all(k in cls for k in ("sensitivity_icbhi", "specificity_icbhi")):
            print(" " * len(head)
                  + f"| [ICBHI Se] : {cls['sensitivity_icbhi']*100:.2f} "
                    f"[ICBHI Sp] : {cls['specificity_icbhi']*100:.2f}  "
                    f"(macro Se/Sp 와 다른 정의)")
        if self.denoise:
            print(" " * len(head)
                  + f"| [SI-SDRi] : {r['si_sdri']:+.2f} dB "
                    f"[NoiseSupp] : {r['noise_suppression_db']:+.2f} dB "
                    f"[SigRetain] : {r['signal_retention_db']:+.2f} dB "
                    f"[Identity] : {r['identity_si_sdr']:+.1f} dB")
        return r, cls
    def run(self):
        save_dir = Path(self.args.save_dir)
        select_on = self.args.select_on
        eval_each = bool(self.args.eval_test_each_epoch) or select_on == "test"
        key = (f"train_cls_{self.args.best_metric}" if select_on == "train"
               else f"cls_{self.args.best_metric}")
        ckpt_path = save_dir / f"{self.args.model_name}_best.pt"
        results, best = [], -1e9
        best_epoch = -1
        for epoch in range(1, self.args.epochs + 1):
            tr = self.train_one_epoch(epoch)
            r = {"epoch": epoch, **tr}
            if eval_each:
                er, _ = self.eval_one_epoch(epoch)
                r.update(er)
            results.append(r)
            if key not in r:
                raise KeyError(
                    f"--best_metric '{self.args.best_metric}' 가 {select_on} 지표에 "
                    f"없습니다. 사용 가능: {sorted(r)}")
            if r[key] > best:
                best, best_epoch = r[key], epoch
                torch.save({"model": self.core.state_dict(), "cfg": self.model_cfg,
                            "args": vars(self.args),
                            "denoise": self.denoise,
                            "noise_aug": self.noise_aug,
                            "epoch": epoch,
                            "select_on": select_on,
                            "best_metric": self.args.best_metric,
                            "best_value": best},
                           ckpt_path)
        pd.DataFrame(results).to_csv(
            save_dir / f"{self.args.model_name}_metrics.csv",
            index=False, encoding="utf-8-sig")
        if best_epoch < 0:
            print("\n[warn] epoch 을 한 번도 돌지 않아 보고할 결과가 없다")
            print(f"저장: {save_dir}")
            return results

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.core.load_state_dict(ckpt["model"])
        print("\n" + "=" * 78)
        print(f"FINAL — {select_on} 기준으로 고른 epoch {best_epoch}"
              f" / {self.args.epochs}  ({self.args.best_metric} = {best*100:.2f})"
              f" 의 체크포인트로 test 1회 평가")
        print("=" * 78)
        _, best_cls = self.eval_one_epoch(best_epoch, tag="TEST")
        scores, labels = self._last_scores
        np.savez_compressed(
            save_dir / f"{self.args.model_name}_best_scores.npz",
            probs=scores, labels=labels, epoch=best_epoch,
            class_names=np.array(self.class_names),
            best_metric=self.args.best_metric, best_value=best)
        print_final_table(best_cls, self.class_names,
                          self.args.model_name, per_class_sp="icbhi")
        csv_path = classification_metrics_to_csv(
            best_cls, save_dir / f"{self.args.model_name}_final_table.csv",
            self.class_names)
        print(f"  표 CSV(논문 붙여넣기용): {csv_path}")

        print(f"저장: {save_dir}")
        return results
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wave_base_path", default=wave_folder)
    p.add_argument("--fold", default=1, type=int,
                   help="train/test dataset에 사용할 grouping 번호")
    p.add_argument("--noise_fold", default=None, type=int,
                   help="noise bank에 사용할 grouping 번호. 기본값은 --fold와 동일")
    p.add_argument("--split_manifest", default=str(DEFAULT_SPLIT_MANIFEST),
                   help="fold,sample_id,split 열을 가진 CSV")
    p.add_argument("--noise_bank", default=None,
                   help="기본값: <data>/db/new_gt/noise_bank_fold<NOISE_FOLD>.npz")
    p.add_argument("--model_name", default="dnet_multiview_v8")
    p.add_argument("--save_dir", default=None,
                   help="기본값: denoising/nytt/ckpt/fold<FOLD>")
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--grad_clip", default=5.0, type=float)
    p.add_argument("--normalize", default="rms",
                   choices=["none", "peak", "rms", "p95"])
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--kernel", default=3, type=int,
                   help="DoubleConv 의 conv kernel 크기. 원논문은 3×3. "
                        "kernel 3 + stride 2 ×4 = 수용영역 200 samples(12.5 ms)")
    p.add_argument("--stride", default=2, type=int,
                   help="다운샘플/업샘플 배율 (MaxPool1d / ConvTranspose1d). 원논문은 2×2")
    p.add_argument("--no_skip", action="store_true",
                   help="ablation 용. 이전 stride-conv 인코더에서는 skip 제거 시 "
                        "F1 42.2→28.2 로 무너졌으나, 이 U-Net 구조에서 같은 결과가 "
                        "재현된다는 보장은 없다 — 직접 다시 측정하라. 용량을 줄이려면 "
                        "--channels 를 우선 쓰십시오.")
    p.add_argument("--channels", default="32,64,128,256,512",
                   help="[enc1,enc2,enc3,enc4,bottleneck] 5개, 채널이 매 단계 2배. "
                        "32,64,128,256,512=2.17M(기본) / 16,32,64,128,256=0.54M / "
                        "16,24,32,64,128=0.15M")
    p.add_argument("--predict", default="signal", choices=["signal", "residual"])

    p.add_argument("--dilations", default="1,2,8,32",
                   help="encoder down block 4개의 dilation (bottleneck은 마지막 값). "
                        "1,1,1,1=기존(11.8 ms) / 1,2,4,16=100 ms 최소 / "
                        "1,2,8,32=권장(206 ms) / 1,4,16,64=407 ms")
    p.add_argument("--num_classes", default=5, type=int)
    p.add_argument("--cls_base", default=32, type=int)

    p.add_argument("--cls_dim", default=32, type=int,
                   help="view별 1x1 conv 출력 채널")

    p.add_argument("--cls_views", default="tonal,transient,square",
                   help="쓸 view 목록. tonal=주파수 보존(지속음) / "
                        "transient=시간 보존(과도음) / square=기존 격자. "
                        "ablation 은 여기서 하나씩 빼면서 한다")
    p.add_argument("--cls_view_pools", default=None,
                   help="view 별 블록 pooling 재정의. 예: "
                        "'tonal=1x2;transient=2x1' 또는 "
                        "'tonal=1x2,1x2,2x2,2x2' (블록 4개분). "
                        "OOM 이면 tonal 뒤쪽 블록의 주파수 pooling 을 2로 올려라")
    p.add_argument("--cls_view_freq_keep", default=None,
                   help="view 별 freq_keep. 정수 하나면 전 view 공통, "
                        "'tonal=16,transient=4,square=4' 형식도 가능. "
                        f"미지정 시 view 기본값 {VIEW_FREQ_KEEP}")
    p.add_argument("--cls_view_nfft", default="auto",
                   help="view 별 mel 분석창(n_fft). 'auto'=권장값(기본) "
                        f"{VIEW_NFFT} / 'tonal=2048,transient=256' 형식으로 개별 지정 / "
                        "'shared' 또는 'none'이면 공용 extractor를 사용. "
                        "hop 은 160(10 ms) 고정이라 view 간 시간축 길이가 맞는다")

    p.add_argument("--cls_view_reduced_dim", default=None, type=int,
                   help=f"view 별 출력 차원 (기본 {VIEW_REDUCED_DIM}). "
                        "예: --cls_view_freq_keep 'tonal=64' 로 해상도를 올렸으면 "
                        "--cls_view_reduced_dim 512 로 압축비 4:1 을 유지한다")

    p.add_argument("--cls_view_dim", default=None,
                   help="view 별 cls_dim. 'tonal=16,transient=32,square=32' 형식. "
                        "미지정 시 --cls_dim 을 전 view 공통으로 쓴다")
    p.add_argument("--cls_view_nmels", default=None,
                   help="view 별 n_mels. 미지정이면 64에서 시작해 n_fft 가 감당할 수 "
                        "있는 수로 자동으로 줄인다(n_fft=256 → 32). "
                        "'tonal=64,transient=32' 형식도 가능")
    p.add_argument("--cls_pool", default="stats", choices=["stats"])
    p.add_argument("--dropout", default=0.1, type=float)

    p.add_argument("--no_denoise", action="store_true",
                   help="denoiser 를 아예 만들지 않는다 (A / A′ arm)")
    p.add_argument("--noise_aug", action=argparse.BooleanOptionalAction, default=None,
                   help="학습 입력에 잡음을 주입할지. 기본 None = denoiser 유무를 "
                        "그대로 따름(이전 동작 유지). A′ arm 은 "
                        "--no_denoise --noise_aug")

    p.add_argument("--mel_normalize", default="minmax", choices=["minmax", "none"])
    p.add_argument("--top_db", default=80.0, type=float)
    p.add_argument("--use_class_weight", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--aux_ce_weight", default=0.0, type=float,
                   help="호환용 인자. v8은 auxiliary logits를 만들지 않아 적용되지 않음")
    p.add_argument("--aux_use_class_weight",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="auxiliary CE에도 class weight를 적용할지")

    p.add_argument("--select_on", default="train", choices=["train", "test"],
                   help="checkpoint 선택에 쓸 split. train=편향 없음(기본) / "
                        "test=선택 편향 발생")

    p.add_argument("--eval_test_each_epoch", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="매 epoch test 도 평가한다(진단용, 기본 꺼짐). "
                        "선택에는 쓰이지 않는다. --select_on test 면 자동으로 켜진다")
    p.add_argument("--best_metric", default="score_icbhi",
                   choices=["sensitivity_macro", "f1_macro", "score_icbhi",
                            "specificity_macro", "accuracy"],
                   help="checkpoint 선택 기준 지표. --select_on 이 가리키는 split "
                        "에서 계산된다. 최종 test 평가는 이렇게 고른 checkpoint "
                        "하나로 1회만 수행한다")
    p.add_argument("--w_rec", default=1.0, type=float)
    p.add_argument("--w_cls", default=1.0, type=float)

    p.add_argument("--snr_min", default=0.0, type=float)
    p.add_argument("--snr_max", default=15.0, type=float)
    p.add_argument("--eval_snr", default=5.0, type=float)
    p.add_argument("--w_l1", default=1.0, type=float)

    p.add_argument("--w_stft", default=0.7, type=float)
    p.add_argument("--w_sisdr", default=0.0, type=float)
    p.add_argument("--stft_fmax_hz", default=2000.0, type=float,
                   help="선형 STFT 손실의 상한 주파수. 0 이하면 전대역(기존 동작). "
                        "폐음 에너지는 거의 전부 2 kHz 아래이고 classifier 도 "
                        "fmax=2000 만 본다.")
    p.add_argument("--stft_ffts", default=None,
                   help="선형 STFT 손실의 n_fft 목록. 미지정 시 대역 제한이면 "
                        f"{BAND_LIMITED_STFT_FFTS}, 전대역이면 {DEFAULT_STFT_FFTS}")
    p.add_argument("--no_amp", action="store_true",
                   help="mixed precision 끄기. 발산·Inf gradient 가 의심되면 사용")

    p.add_argument("--fix_eval_rng", action="store_true",
                   help="eval 의 전역 torch.manual_seed 호출을 없애고 noise bank "
                        "rng 만 저장/복원한다. epoch 마다 shuffle·SNR·잡음이 "
                        "실제로 달라진다. arm 간 비교 시 전부 같은 설정이어야 함")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num_workers", default=5, type=int)
    p.add_argument("--max_train_batches", default=None, type=int)
    p.add_argument("--max_eval_batches", default=None, type=int)
    p.add_argument("--gpus", default=None,
               help="사용할 물리 GPU id, 콤마 구분. 예: 5,6,7. 미지정 시 전체 노출 GPU 사용")
    args = p.parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.fold < 1:
        p.error("--fold는 1 이상의 정수여야 합니다.")
    if args.noise_fold is None:
        args.noise_fold = args.fold
    if args.noise_fold < 1:
        p.error("--noise_fold는 1 이상의 정수여야 합니다.")
    if args.aux_ce_weight < 0:
        p.error("--aux_ce_weight는 0 이상이어야 합니다.")
    if args.noise_bank is None:
        args.noise_bank = str(
            DEFAULT_NOISE_BANK_DIR / f"noise_bank_fold{args.noise_fold}.npz"
        )
    if args.save_dir is None:
        args.save_dir = str(CURRENT_DIR.parent / "ckpt" / f"fold{args.fold}")
    return args

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    Trainer(args).run()
