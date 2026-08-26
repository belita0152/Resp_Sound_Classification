"""
    Create AUROC and AUPRC curves.

    python -m denoising.utils.roc_pr --scores ckpt/best_scores.npz  --> 단일 auroc, auprc curve 출력
    python -m denoising.utils.roc_pr --scores fold1.npz fold2.npz fold3.npz  --> 평균 auroc, auprc 출력

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = ["#1f4e79", "#c1121f", "#2a9d8f", "#e07a1f", "#6a4c93"]


def _thresholds(scores):
    order = np.argsort(-scores, kind="mergesort")
    indices = np.r_[np.flatnonzero(np.diff(scores[order])), scores.size - 1]
    return order, indices


def roc_curve(y_true, scores):
    """Return false-positive rate, true-positive rate, and AUROC."""
    order, indices = _thresholds(scores)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    positives, negatives = tp[-1], fp[-1]
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    tpr = np.r_[0.0, tp[indices] / positives]
    fpr = np.r_[0.0, fp[indices] / negatives]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fpr, tpr, float(integrate(tpr, fpr))


def pr_curve(y_true, scores):
    """Return recall, precision, and average precision."""
    order, indices = _thresholds(scores)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    positives = tp[-1]
    if positives == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan")

    precision = tp[indices] / (tp[indices] + fp[indices])
    recall = tp[indices] / positives
    average_precision = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return np.r_[0.0, recall], np.r_[1.0, precision], float(average_precision)


def evaluate(probs, labels, n_classes):
    result = {"per_class": [], "curves": {}}
    aurocs, auprcs = [], []

    for class_index in range(n_classes):
        y_true = (labels == class_index).astype(np.int64)
        fpr, tpr, auroc = roc_curve(y_true, probs[:, class_index])
        recall, precision, auprc = pr_curve(y_true, probs[:, class_index])
        prevalence = float(y_true.mean())
        metrics = {
            "cls": class_index,
            "n": int(y_true.sum()),
            "prevalence": prevalence,
            "auroc": auroc,
            "auprc": auprc,
            "auprc_lift": auprc / prevalence if prevalence else float("nan"),
        }
        result["per_class"].append(metrics)
        result["curves"][class_index] = (fpr, tpr, recall, precision)
        if np.isfinite(auroc):
            aurocs.append(auroc)
        if np.isfinite(auprc):
            auprcs.append(auprc)

    one_hot = np.eye(n_classes, dtype=np.int64)[labels].ravel()
    fpr, tpr, micro_auroc = roc_curve(one_hot, probs.ravel())
    recall, precision, micro_auprc = pr_curve(one_hot, probs.ravel())
    result["macro"] = {"auroc": float(np.mean(aurocs)), "auprc": float(np.mean(auprcs))}
    result["micro"] = {"auroc": micro_auroc, "auprc": micro_auprc}
    result["curves"]["micro"] = (fpr, tpr, recall, precision)
    return result


def _style(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))


def fig_roc_pr(result, names, output, title=""):
    figure, (roc_axes, pr_axes) = plt.subplots(1, 2, figsize=(10.6, 4.6))
    roc_axes.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="chance (0.500)")

    for index, name in enumerate(names):
        fpr, tpr, recall, precision = result["curves"][index]
        metrics = result["per_class"][index]
        color = COLORS[index % len(COLORS)]
        roc_axes.plot(
            fpr, tpr, color=color, lw=1.8,
            label=f"{name} (n={metrics['n']}) AUROC {metrics['auroc']:.3f}",
        )
        pr_axes.plot(
            recall, precision, color=color, lw=1.8,
            label=f"{name} AP {metrics['auprc']:.3f} (base {metrics['prevalence']:.3f})",
        )
        pr_axes.axhline(metrics["prevalence"], color=color, ls=":", lw=0.9, alpha=0.55)

    roc_axes.set(
        xlabel="False positive rate (1 - Specificity)",
        ylabel="True positive rate (Sensitivity)",
        title=f"ROC - one-vs-rest    macro {result['macro']['auroc']:.3f}",
    )
    pr_axes.set(
        xlabel="Recall (Sensitivity)",
        ylabel="Precision",
        title=f"Precision-Recall    macro AP {result['macro']['auprc']:.3f}",
    )
    for axes, location in ((roc_axes, "lower right"), (pr_axes, "upper right")):
        axes.legend(fontsize=7.5, loc=location, framealpha=0.92)
        _style(axes)

    if title:
        figure.suptitle(title, fontsize=10.5, x=0.01, ha="left")
        figure.tight_layout(rect=[0, 0, 1, 0.955])
    else:
        figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def load(path):
    with np.load(path, allow_pickle=True) as data:
        probs = data["probs"].astype(np.float64)
        labels = data["labels"].astype(np.int64)
        names = [str(value) for value in data["class_names"].tolist()]
        epoch = int(data["epoch"]) if "epoch" in data.files else -1
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"Invalid probabilities: {path}")
    return probs, labels, names, epoch


def _print_summary(run_name, path, epoch, labels, names, result):
    print(f"\n{run_name} ({path.name}, epoch {epoch}, n={len(labels):,})")
    print(f"{'class':10}{'n':>6}{'prev':>9}{'AUROC':>9}{'AUPRC':>9}{'lift':>9}")
    for index, name in enumerate(names):
        metrics = result["per_class"][index]
        print(f"{name:10}{metrics['n']:>6}{metrics['prevalence']:>9.3f}"
              f"{metrics['auroc']:>9.3f}{metrics['auprc']:>9.3f}"
              f"{metrics['auprc_lift']:>8.2f}x")
    for average in ("macro", "micro"):
        metrics = result[average]
        print(f"{average:10}{'':>15}{metrics['auroc']:>9.3f}{metrics['auprc']:>9.3f}")


def _save_json(path, run_name, epoch, result):
    payload = {"run": run_name, "epoch": epoch,
               "per_class": result["per_class"],
               "macro": result["macro"], "micro": result["micro"]}
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=float)


def _print_average(results):
    if len(results) < 2:
        return
    mean_auroc = np.mean([result["macro"]["auroc"] for result in results])
    mean_auprc = np.mean([result["macro"]["auprc"] for result in results])
    print(f"\nAverage across {len(results)} files")
    print(f"  macro AUROC: {mean_auroc:.3f}")
    print(f"  macro AUPRC: {mean_auprc:.3f}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--out-dir", "--out_dir", default="figures")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_paths = [Path(path) for path in args.scores]
    results, rows = [], []

    for index, path in enumerate(score_paths, start=1):
        run_name = path.stem.replace("_best_scores", "")
        probs, labels, names, epoch = load(path)
        result = evaluate(probs, labels, len(names))
        results.append(result)
        _print_summary(run_name, path, epoch, labels, names, result)

        for name, metrics in zip(names, result["per_class"]):
            rows.append({"run": run_name, "epoch": epoch, "cls": name,
                         **{key: value for key, value in metrics.items() if key != "cls"}})

        output_name = f"{index}_{run_name}" if len(score_paths) > 1 else run_name
        fig_roc_pr(result, names, output_dir / f"roc_pr_{output_name}.png",
                   title=f"{run_name} - best epoch {epoch}")
        _save_json(output_dir / f"roc_pr_{output_name}.json", run_name, epoch, result)

    pd.DataFrame(rows).to_csv(output_dir / "roc_pr_summary.csv",
                              index=False, encoding="utf-8-sig")
    _print_average(results)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
