# -*- coding:utf-8 -*-
import numpy as np
from typing import Dict, List, Optional


def calculate_segmentation_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
    epsilon: float = 1e-6
) -> Dict[str, float]:

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    mask = (targets != ignore_index)
    preds_masked = preds[mask]
    targets_masked = targets[mask]

    conf_matrix = np.bincount(
        num_classes * targets_masked.astype(np.int32) + preds_masked.astype(np.int32),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).astype(np.float32)

    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp

    accuracy = tp.sum() / (conf_matrix.sum() + epsilon)

    per_class_iou = tp / (tp + fp + fn + epsilon)
    per_class_dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    per_class_precision = tp / (tp + fp + epsilon)
    per_class_recall = tp / (tp + fn + epsilon)

    micro_iou = tp.sum() / (tp.sum() + fp.sum() + fn.sum() + epsilon)
    micro_dice = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + epsilon)
    macro_iou = per_class_iou.mean()
    macro_dice = per_class_dice.mean()

    support = conf_matrix.sum(axis=1)
    weighted_iou = (per_class_iou * support / (support.sum() + epsilon)).sum()
    weighted_dice = (per_class_dice * support / (support.sum() + epsilon)).sum()

    return {
        "accuracy": accuracy.item(),
        "iou_micro": micro_iou.item(),
        "iou_macro": macro_iou.item(),
        "iou_weighted": weighted_iou.item(),
        "dice_micro": micro_dice.item(),
        "dice_macro": macro_dice.item(),
        "dice_weighted": weighted_dice.item(),
        "per_class_iou": per_class_iou.tolist(),
        "per_class_dice": per_class_dice.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
    }


def calculate_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
    normal_class: int = 0,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """segment 단위 분류 지표.

    accuracy    : 전체 정확도
    sensitivity : per-class recall = TP / (TP + FN)
    specificity : per-class one-vs-rest = TN / (TN + FP)
    f1          : 2·P·R / (P + R)

    ICBHI 2017 공식 지표도 함께 계산한다 (선행연구 비교용).
        Se_icbhi = 이상음 클래스들의 micro recall  (정상 제외)
        Sp_icbhi = 정상 클래스의 recall
        Score    = (Se_icbhi + Sp_icbhi) / 2
      ※ Se_icbhi 는 크기 가중 micro 평균이므로 최소 클래스의 비중이 작다.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    mask = ((targets != ignore_index) if ignore_index is not None
            else np.ones_like(targets, bool))
    preds_masked = preds[mask].astype(np.int32)
    targets_masked = targets[mask].astype(np.int32)

    conf_matrix = np.bincount(
        num_classes * targets_masked + preds_masked,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).astype(np.float64)

    total = conf_matrix.sum()
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    tn = total - tp - fp - fn
    support = conf_matrix.sum(axis=1)

    accuracy = tp.sum() / (total + epsilon)

    per_class_sensitivity = tp / (tp + fn + epsilon)          # = recall
    per_class_specificity = tn / (tn + fp + epsilon)
    per_class_precision = tp / (tp + fp + epsilon)
    per_class_f1 = (2 * per_class_precision * per_class_sensitivity
                    / (per_class_precision + per_class_sensitivity + epsilon))

    present = support > 0          # 등장하지 않은 클래스는 macro 에서 제외
    macro_sensitivity = per_class_sensitivity[present].mean() if present.any() else 0.0
    macro_specificity = per_class_specificity[present].mean() if present.any() else 0.0
    macro_precision = per_class_precision[present].mean() if present.any() else 0.0
    macro_f1 = per_class_f1[present].mean() if present.any() else 0.0
    weighted_f1 = (per_class_f1 * support / (support.sum() + epsilon)).sum()

    abnormal = [c for c in range(num_classes) if c != normal_class]
    se_icbhi = tp[abnormal].sum() / (support[abnormal].sum() + epsilon)
    sp_icbhi = tp[normal_class] / (support[normal_class] + epsilon)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "sensitivity_macro": float(macro_sensitivity),
        "specificity_macro": float(macro_specificity),
        "precision_macro": float(macro_precision),
        "per_class_sensitivity": per_class_sensitivity.tolist(),
        "per_class_specificity": per_class_specificity.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "support": support.astype(int).tolist(),
        "confusion_matrix": conf_matrix.astype(int).tolist(),
        "se_icbhi": float(se_icbhi),
        "sp_icbhi": float(sp_icbhi),
        "score_icbhi": float((se_icbhi + sp_icbhi) / 2),
    }


def print_classification_metrics(result: Dict, class_names: Optional[List] = None) -> None:
    n = len(result["support"])
    names = class_names or [str(i) for i in range(n)]

    print(f"Accuracy {result['accuracy']*100:.2f}  "
          f"F1(macro) {result['f1_macro']*100:.2f}  "
          f"Se(macro) {result['sensitivity_macro']*100:.2f}  "
          f"Sp(macro) {result['specificity_macro']*100:.2f}")
    print(f"ICBHI  Se {result['se_icbhi']*100:.2f}  "
          f"Sp {result['sp_icbhi']*100:.2f}  "
          f"Score {result['score_icbhi']*100:.2f}")

    print(f"{'class':12}{'n':>7}{'Se':>8}{'Sp':>8}{'Prec':>8}{'F1':>8}")
    for i in range(n):
        print(f"{names[i][:11]:12}{result['support'][i]:7d}"
              f"{result['per_class_sensitivity'][i]*100:8.2f}"
              f"{result['per_class_specificity'][i]*100:8.2f}"
              f"{result['per_class_precision'][i]*100:8.2f}"
              f"{result['per_class_f1'][i]*100:8.2f}")

    print("confusion matrix (row=true, col=pred)")
    for i, row in enumerate(result["confusion_matrix"]):
        print(f"  {names[i][:11]:12}" + "".join(f"{v:7d}" for v in row))
