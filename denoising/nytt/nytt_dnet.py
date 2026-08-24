# -*- coding: utf-8 -*-
"""NyTT-DNet command-line entry point.

학습 코드는 ``trainer.py``, 모델은 ``module.py``, 지표는 ``denoising_metric.py``,
데이터/split 보조 코드는 ``utils.py``에 둔다.
"""
from __future__ import annotations

import argparse
import warnings

import random
import numpy as np
import torch

from .trainer import Trainer
from .utils import (CURRENT_DIR, DEFAULT_NOISE_BANK_DIR,
                    DEFAULT_SPLIT_MANIFEST, wave_folder)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--wave_base_path", default=wave_folder)
    p.add_argument("--fold", default=1, type=int,
                   help="split_manifest에서 사용할 grouping 번호")
    p.add_argument("--split_manifest", default=str(DEFAULT_SPLIT_MANIFEST),
                   help="fold,sample_id,split 열을 가진 CSV")
    p.add_argument("--noise_bank", default=None,
                   help="기본값: <data>/db/new_gt/noise_bank_fold<FOLD>.npz")
    p.add_argument("--model_name", default="dnet_multiscale")
    p.add_argument("--save_dir", default=None,
                   help="기본값: denoising/nytt/ckpt/fold<FOLD>")

    # train
    p.add_argument("--epochs", default=25, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--grad_clip", default=5.0, type=float)
    p.add_argument("--normalize", default="rms",
                   choices=["none", "peak", "rms", "p95"])
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no_denoise", action="store_true",
                   help="A control: no denoise, no noise augmentation")
    p.add_argument(
        "--no_noise_augmentation", "--no_noise_aug",
        dest="no_noise_augmentation", action="store_true",
        help=("A-prime arm: DenoisingUNet은 사용하지만 학습 입력에 추가 noise를 "
              "주입하지 않는다."),
    )

    # model — Denoising UNet
    p.add_argument("--kernel", default=3, type=int,
                   help="DoubleConv 의 conv kernel 크기. 원논문은 3×3")
    p.add_argument("--stride", default=2, type=int,
                   help="다운/업샘플 배율 (MaxPool1d / ConvTranspose1d)")
    p.add_argument("--no_skip", action="store_true", help="skip 연결 제거 ablation")
    p.add_argument("--channels", default="32,64,128,256,512",
                   help="[enc1,enc2,enc3,enc4,bottleneck] 5개"
                        "32,64,128,256,512=2.17M(기본) / 16,32,64,128,256=0.54M / "
                        "16,24,32,64,128=0.15M")
    p.add_argument("--predict", default="signal", choices=["signal", "residual"])

    # classifier
    p.add_argument("--num_classes", default=5, type=int)
    p.add_argument("--cls_base", default=32, type=int)
    p.add_argument("--cls_head", default="multiscale",
                   choices=["multiscale", "plain"])
    p.add_argument("--cls_scales", default="1,2,3,4",
                   help="쓸 블록 번호. 1=20ms/f, 2=40ms/f, 3=80ms/f, 4=160ms/f")
    p.add_argument("--cls_dim", default=32, type=int,
                   help="스케일별 1x1 conv 출력 채널")
    p.add_argument("--cls_freq_keep", default=4, type=int,
                   help="스케일별로 남길 주파수 칸 수")
    p.add_argument("--cls_pool", default="attn", choices=["avg", "max", "attn"])
    p.add_argument("--dropout", default=0.1, type=float)

    p.add_argument("--mel_normalize", default="minmax", choices=["minmax", "none"])
    p.add_argument("--top_db", default=80.0, type=float)
    p.add_argument("--use_class_weight", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument(
        "--best_metric", default="score_icbhi",
        choices=["sensitivity_macro", "f1_macro", "score_icbhi",
                 "specificity_macro", "accuracy"],
        help=("clean train inference 성능으로 checkpoint를 고르는 기준. "
              "test는 선택에 사용하지 않고 best checkpoint에서 한 번만 평가한다."),
    )
    p.add_argument("--verbose_metrics", action="store_true")
    p.add_argument("--w_rec", default=1.0, type=float)
    p.add_argument("--w_cls", default=1.0, type=float)

    # NyTT
    p.add_argument("--snr_min", default=0.0, type=float)
    p.add_argument("--snr_max", default=15.0, type=float)
    p.add_argument("--eval_snr", default=5.0, type=float)
    p.add_argument("--w_l1", default=1.0, type=float)
    p.add_argument("--w_stft", default=1.0, type=float)
    p.add_argument("--w_sisdr", default=0.0, type=float)

    p.add_argument("--no_amp", action="store_true",
                   help="mixed precision 끄기")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num_workers", default=5, type=int)
    p.add_argument("--max_train_batches", default=None, type=int)

    # args = p.parse_args(argv)
    # if args.fold < 1:
    #     p.error("--fold는 1 이상의 정수여야 합니다.")
    # if args.epochs < 1:
    #     p.error("--epochs는 1 이상의 정수여야 합니다.")
    # if args.noise_bank is None:
    #     args.noise_bank = str(DEFAULT_NOISE_BANK_DIR / f"noise_bank_fold{args.fold}.npz")
    # if args.save_dir is None:
    #     args.save_dir = str(CURRENT_DIR / "ckpt" / f"fold{args.fold}")
    # return args


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = get_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    Trainer(args).run()


if __name__ == "__main__":
    main()
