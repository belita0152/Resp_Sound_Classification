# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .loss import TrainingLoss
from .denoising_metric import (calculate_classification_metrics, energy_ratio_db,
                               mix_at_snr, print_classification_metrics, si_sdr)
from .module import DenoisingUNet, DNetClassifier
from .utils import (build_fold_wave_datasets, load_noise_bank, print_run_config,
                    resolve_arm)


class Trainer(object):
    # 서브클래스가 갈아끼울 수 있게 둔다 (nytt_binary.py 가 라벨을 재매핑한다)
    CLASS_NAMES = ["Normal", "Stridor", "Rhonchi", "Wheezing", "Crackle"]

    def build_datasets(self, args):
        """(train_ds, test_ds)를 만든다. 라벨 체계를 바꾸려면 이 메서드만 재정의한다."""
        return build_fold_wave_datasets(args)

    def __init__(self, args):
        self.args = args
        print_run_config(args)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        train_ds, test_ds = self.build_datasets(args)
        print(f"[data] train {len(train_ds):,} segments / {len(train_ds.sample_ids)} ids")
        print(f"[data] test  {len(test_ds):,} segments / {len(test_ds.sample_ids)} ids")

        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True, num_workers=args.num_workers,
                                       pin_memory=self.device.type == "cuda")

        self.train_eval_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=self.device.type == "cuda",
        )

        self.test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=self.device.type == "cuda",
        )

        self.arm, self.use_dnet, self.noise_augmentation = resolve_arm(
            args.no_denoise, args.no_noise_augmentation,
        )
        self.noise_bank = load_noise_bank(
            args, train_ds.sample_ids, test_ds.sample_ids,
        )
        self.evaluate_denoising = self.noise_bank is not None
        print(f"[arm] {self.arm}: dnet={'on' if self.use_dnet else 'off'}  "
              f"noise_augmentation={'on' if self.noise_augmentation else 'off'}  "
              f"denoising_metrics={'on' if self.evaluate_denoising else 'off'}")

        # Model
        channels = tuple(int(value) for value in args.channels.split(",") if value.strip())

        self.dnet_cfg = {
            "in_channels": 1,
            "channels": channels,
            "kernel": args.kernel,
            "stride": args.stride,
            "use_skip": not args.no_skip,
            "predict": args.predict,
        }

        dnet = DenoisingUNet(**self.dnet_cfg) if self.use_dnet else None
        self.base_model = DNetClassifier(
            dnet=dnet,
            num_classes=args.num_classes,
            base=args.cls_base,
            dropout=args.dropout,
            mel_normalize=args.mel_normalize,
            top_db=args.top_db,
            cls_head=args.cls_head,
            cls_scales=tuple(int(value) for value in args.cls_scales.split(",")
                             if value.strip()),
            cls_dim=args.cls_dim,
            cls_freq_keep=args.cls_freq_keep,
            cls_pool=args.cls_pool,
        ).to(self.device)
        self.model: nn.Module = self.base_model

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.base_model)

        n_all = sum(p.numel() for p in self.model.parameters())
        if dnet is None:
            print(f"[model] 인코더 없음   총 {n_all/1e6:.2f} M")
        else:
            n_dnet = sum(p.numel() for p in dnet.parameters())
            print(f"[model] dnet {n_dnet/1e6:.2f} M / total {n_all/1e6:.2f} M  "
                  f"use_skip={not args.no_skip}  predict={args.predict}")

        self.class_names = list(self.CLASS_NAMES)[:args.num_classes]

        # Loss
        w = train_ds.class_weights(args.num_classes).to(self.device)
        self.loss = TrainingLoss(
            class_weights=w if args.use_class_weight else None,
            use_dnet=self.use_dnet,
            w_rec=args.w_rec,
            w_cls=args.w_cls,
            w_l1=args.w_l1,
            w_stft=args.w_stft,
            w_sisdr=args.w_sisdr,
        ).to(self.device)

        print(f"[cls] train counts {train_ds.class_counts}")
        print(f"[cls] class_weight={'on' if args.use_class_weight else 'off'} "
              f"{[round(v, 3) for v in w.tolist()]}")
        print(f"[cls] head={args.cls_head}  pool={args.cls_pool}"
              + (f"  scales={args.cls_scales}  dim={args.cls_dim}"
                 f"  freq_keep={args.cls_freq_keep}"
                 if args.cls_head == "multiscale" else "")
              + f"   ({sum(p.numel() for p in self.base_model.classifier.parameters())/1e3:.1f} K)")
        print(f"[norm] ① 파형 입력 {args.normalize}  ② denoiser 출력 rms 정합 (모델 내부)")
        print(f"[norm] ③ mel {args.mel_normalize} (top_db={args.top_db}) "
              f"← A arm(data_loader MelTransform) 과 동일해야 함")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=args.epochs)
        self.amp = (self.device.type == "cuda") and not args.no_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    def _make_pair(self, x_target, snr_db=None):
        b, _, t = x_target.shape
        if self.noise_bank is None:
            raise RuntimeError("noise bank가 없어 noisy pair를 만들 수 없습니다.")
        noise = torch.as_tensor(self.noise_bank.sample(t, batch=b),
                                dtype=torch.float32,
                                device=x_target.device).unsqueeze(1)
        if snr_db is None:
            snr_db = (torch.rand(b, device=x_target.device)
                      * (self.args.snr_max - self.args.snr_min) + self.args.snr_min)
        return mix_at_snr(x_target, noise, snr_db), noise

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total, total_rec, total_cls, n = 0.0, 0.0, 0.0, 0

        for batch_idx, (data, target) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad(set_to_none=True)
            x_target = data.to(torch.float32).to(self.device, non_blocking=True)
            y = target.long().to(self.device, non_blocking=True)

            if self.noise_augmentation:
                with torch.no_grad():
                    x_in, _ = self._make_pair(x_target)
            else:
                x_in = x_target          # A and A': no noise addition

            with torch.cuda.amp.autocast(enabled=self.amp):
                x_hat, logits = self.model(x_in)
                loss, loss_rec, loss_cls = self.loss(x_hat, x_target, logits, y)

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            if self.args.grad_clip and self.args.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = x_target.size(0)
            total += float(loss.detach()) * bs
            total_rec += float(loss_rec.detach()) * bs
            total_cls += float(loss_cls.detach()) * bs
            n += bs
            if self.args.max_train_batches and batch_idx >= self.args.max_train_batches:
                break

        self.scheduler.step()
        n = max(n, 1)
        return {"train_loss": total / n, "train_rec": total_rec / n,
                "train_cls": total_cls / n}

    @torch.inference_mode()
    def evaluate(self, loader, split: str, epoch: int | None = None,
                 include_denoising: bool = False, verbose: bool = False):
        self.model.eval()
        acc = {"cls_loss": 0.0}
        if include_denoising and self.evaluate_denoising:
            acc.update(si_sdr_in=0.0, si_sdr_out=0.0, l1=0.0,
                       identity_si_sdr=0.0)
        supp, retain, n_count, n_batch = 0.0, 0.0, 0, 0
        preds_all, reals_all = [], []

        dnet = self.base_model.dnet
        if include_denoising and self.evaluate_denoising:
            self.noise_bank.rng = np.random.default_rng(self.args.seed)

        for batch_idx, (data, target) in enumerate(loader, start=1):
            x_target = data.to(torch.float32).to(self.device, non_blocking=True)
            y = target.long().to(self.device, non_blocking=True)

            x_clean_hat, logits = self.model(x_target)

            bs = x_target.size(0)
            acc["cls_loss"] += float(self.loss.classification(logits, y)) * bs
            n_count += bs
            preds_all.append(logits.argmax(dim=1).detach().cpu())
            reals_all.append(y.detach().cpu())

            if include_denoising and self.evaluate_denoising:
                snr = torch.full((bs,), self.args.eval_snr, device=self.device)
                x_in, noise = self._make_pair(x_target, snr_db=snr)
                x_hat, _ = self.model(x_in)

                acc["si_sdr_in"] += si_sdr(x_in, x_target).mean().item() * bs
                acc["si_sdr_out"] += si_sdr(x_hat, x_target).mean().item() * bs
                acc["l1"] += (x_hat - x_target).abs().mean().item() * bs
                supp += energy_ratio_db(dnet, noise)           # the larger, the btter
                retain += -energy_ratio_db(dnet, x_target)     # the closer to 0, the better
                # ★ Identity — > 20dB : denoiser not functions / 10dB : denoiser functions well
                acc["identity_si_sdr"] += (si_sdr(x_clean_hat, x_target).mean().item() * bs)
                n_batch += 1

        r = {k: v / max(n_count, 1) for k, v in acc.items()}
        if include_denoising and self.evaluate_denoising:
            r["si_sdri"] = r["si_sdr_out"] - r["si_sdr_in"]
            r["noise_suppression_db"] = supp / max(n_batch, 1)
            r["signal_retention_db"] = retain / max(n_batch, 1)

        y_pred = torch.cat(preds_all).numpy()
        y_true = torch.cat(reals_all).numpy()
        cls = calculate_classification_metrics(y_pred, y_true,
                                               num_classes=self.args.num_classes)
        r.update({f"cls_{k}": v for k, v in cls.items() if not isinstance(v, list)})

        where = f" epoch {epoch:03d}" if epoch is not None else ""
        head = f"[{split}{where}] => "
        print(head
              + f"[Acc] : {cls['accuracy']*100:.2f} "
                f"[F1] : {cls['f1_macro']*100:.2f} "
                f"[Se] : {cls['sensitivity_macro']*100:.2f} "
                f"[Sp] : {cls['specificity_macro']*100:.2f} "
                f"[ICBHI] : {cls['score_icbhi']*100:.2f}")
        # denoiser 가 없으면 denoising 줄 자체를 찍지 않는다
        if include_denoising and self.evaluate_denoising:
            print(" " * len(head)
                  + f"| [SI-SDRi] : {r['si_sdri']:+.2f} dB "
                    f"[NoiseSupp] : {r['noise_suppression_db']:+.2f} dB "
                    f"[SigRetain] : {r['signal_retention_db']:+.2f} dB "
                    f"[Identity] : {r['identity_si_sdr']:+.1f} dB")

        per_class = " | ".join(
            f"{self.class_names[i][:3]}(n={cls['support'][i]}) "
            f"{cls['per_class_sensitivity'][i]*100:.1f}/"
            f"{cls['per_class_specificity'][i]*100:.1f}/"
            f"{cls['per_class_f1'][i]*100:.1f}"
            for i in range(self.args.num_classes)
        )
        print(" " * len(head) + f"| PerClass Se/Sp/F1  {per_class}")

        if verbose or self.args.verbose_metrics:
            print_classification_metrics(cls, self.class_names)
        return r, cls

    def run(self):
        save_dir = Path(self.args.save_dir)
        args_path = save_dir / f"{self.args.model_name}_args.json"
        metrics_path = save_dir / f"{self.args.model_name}_metrics.csv"
        checkpoint_path = save_dir / f"{self.args.model_name}_best.pt"
        final_path = save_dir / f"{self.args.model_name}_best_metrics.json"

        with open(args_path, "w", encoding="utf-8") as fp:
            json.dump(vars(self.args), fp, ensure_ascii=False, indent=2, default=str)

        if self.args.max_eval_batches:
            print("[debug] --max_eval_batches가 설정되어 TrainEval/Test가 일부 batch만 "
                  "사용합니다. 이 test 수치는 공식 최종 성능으로 쓰면 안 됩니다.")

        results = []
        best_value = -float("inf")
        best_epoch = -1
        best_train_cls = None

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_one_epoch()
            train_eval, train_cls = self.evaluate(
                self.train_eval_loader,
                split="TrainEval",
                epoch=epoch,
                include_denoising=False,
                verbose=False,
            )

            row = {"epoch": epoch, **train_loss}
            row.update({f"train_eval_{key}": value for key, value in train_eval.items()})
            results.append(row)

            selection_value = float(train_cls[self.args.best_metric])
            if selection_value > best_value:
                best_value = selection_value
                best_epoch = epoch
                best_train_cls = train_cls
                torch.save(
                    {
                        "model": self.base_model.state_dict(),
                        "dnet_cfg": self.dnet_cfg,
                        "args": vars(self.args),
                        "epoch": epoch,
                        "best_metric": self.args.best_metric,
                        "best_value": best_value,
                        "selection_split": "train",
                        "selection_input": "clean",
                        "selection_model_mode": "eval",
                        "arm": self.arm,
                        "use_dnet": self.use_dnet,
                        "noise_augmentation": self.noise_augmentation,
                        "denoising_metrics": self.evaluate_denoising,
                        "train_classification": train_cls,
                    },
                    checkpoint_path,
                )

        pd.DataFrame(results).to_csv(metrics_path, index=False, encoding="utf-8-sig")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False,
        )
        self.base_model.load_state_dict(checkpoint["model"])
        print(f"[checkpoint] epoch {best_epoch}/{self.args.epochs}, "
              f"train {self.args.best_metric}={best_value * 100:.2f}")


        # Test Results -------------------------------------------------------
        test_metrics, test_cls = self.evaluate(
            self.test_loader,
            split="Test",
            epoch=best_epoch,
            include_denoising=True,
            verbose=False,
        )

        print("\nTest Metrics:")
        print_classification_metrics(test_cls, self.class_names)

        denoising_keys = (
            "si_sdri", "noise_suppression_db", "signal_retention_db",
            "identity_si_sdr",
        )

        test_denoising = {
            key: test_metrics.get(key) for key in denoising_keys
            if key in test_metrics
        }

        report = {
            "best_epoch": best_epoch,
            "best_metric": self.args.best_metric,
            "best_value": best_value,
            "arm": self.arm,
            "use_dnet": self.use_dnet,
            "noise_augmentation": self.noise_augmentation,
            "denoising_metrics": self.evaluate_denoising,
            "selection": {
                "split": "train",
                "input": "clean",
                "model_mode": "eval",
                "classification": best_train_cls,
            },
            "test": {
                "evaluation_count": 1,
                "classification": test_cls,
                "denoising": test_denoising,
            },
            # 기존 결과 분석 코드와의 호환을 위한 최종 test 별칭이다.
            "classification": test_cls,
            "denoising": test_denoising,
        }
        with open(final_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2, default=str)

        last = results[-1]
        print(f"\n[참고: train 성능] 마지막 epoch {last['epoch']} : "
              f"Acc {last['train_eval_cls_accuracy'] * 100:.2f} "
              f"F1 {last['train_eval_cls_f1_macro'] * 100:.2f} "
              f"Se {last['train_eval_cls_sensitivity_macro'] * 100:.2f} "
              f"ICBHI {last['train_eval_cls_score_icbhi'] * 100:.2f}")
        print(f"저장: {save_dir}")
        return results
