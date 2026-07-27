from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_coco_threshold_analysis(coco_dir, output_dir, prefix="coco"):
    """Plots derived directly from raw COCOeval arrays (precision/recall/scores).

    Nothing here reimplements COCO's matching logic; it only reads and reshapes the
    arrays that pycocotools already computed, so numbers stay identical to the official
    AP/AR reported by COCOEvaluator. `coco_dir` must contain precision.npy, recall.npy,
    scores.npy, iou_thresholds.npy and recall_thresholds.npy, as saved by an
    evaluate_coco.py-style script. Returns a dict of scalar summary numbers so callers
    can merge it into their own summary.json.
    """

    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # precision: [T, R, K, A, M]  recall: [T, K, A, M]  scores: [T, R, K, A, M]
    precision = np.load(coco_dir / "precision.npy")
    recall = np.load(coco_dir / "recall.npy")
    scores = np.load(coco_dir / "scores.npy")
    iou_thresholds = np.load(coco_dir / "iou_thresholds.npy")
    recall_thresholds = np.load(coco_dir / "recall_thresholds.npy")

    CAT = 0
    AREA = 0
    MAX_DETS = -1

    IOU_50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))
    IOU_75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))

    # ============================
    # PR curves per IoU threshold
    # ============================

    plt.figure(figsize=(7, 6))

    mean_precision = precision[:, :, CAT, AREA, MAX_DETS].mean(axis=0)
    mean_precision = np.where(mean_precision < 0, np.nan, mean_precision)

    for idx, label in [(IOU_50, "IoU=0.50"), (IOU_75, "IoU=0.75")]:
        p = precision[idx, :, CAT, AREA, MAX_DETS]
        p = np.where(p < 0, np.nan, p)
        plt.plot(recall_thresholds, p, label=label)

    plt.plot(recall_thresholds, mean_precision, "--", label="mean IoU=0.50:0.95", color="black")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("COCOeval Precision-Recall Curve")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_pr_curves.png")
    plt.close()

    # ============================
    # Precision (AP) vs IoU threshold
    # ============================

    ap_per_iou = np.array([
        np.nanmean(np.where(precision[t, :, CAT, AREA, MAX_DETS] < 0, np.nan, precision[t, :, CAT, AREA, MAX_DETS]))
        for t in range(len(iou_thresholds))
    ])

    plt.figure(figsize=(7, 5))
    plt.plot(iou_thresholds, ap_per_iou, marker="o")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Precision (AP)")
    plt.title("COCOeval Precision vs IoU Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_precision_vs_iou.png")
    plt.close()

    # ============================
    # Recall (AR) vs IoU threshold
    # ============================

    ar_per_iou = recall[:, CAT, AREA, MAX_DETS]

    plt.figure(figsize=(7, 5))
    plt.plot(iou_thresholds, ar_per_iou, marker="o", color="tab:orange")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Recall (AR)")
    plt.title("COCOeval Recall vs IoU Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_recall_vs_iou.png")
    plt.close()

    # ============================
    # Precision / Recall vs Confidence Threshold
    # (reconstructed from COCOeval's internal score array, at fixed IoU)
    # ============================

    def precision_recall_vs_score(iou_idx):

        p = precision[iou_idx, :, CAT, AREA, MAX_DETS]
        s = scores[iou_idx, :, CAT, AREA, MAX_DETS]

        valid = p >= 0

        p = p[valid]
        r = recall_thresholds[valid]
        s = s[valid]

        order = np.argsort(s)

        return s[order], p[order], r[order]

    plt.figure(figsize=(7, 5))

    for idx, label in [(IOU_50, "IoU=0.50"), (IOU_75, "IoU=0.75")]:

        s, p, r = precision_recall_vs_score(idx)

        plt.plot(s, p, label=f"Precision ({label})")
        plt.plot(s, r, "--", label=f"Recall ({label})")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric value")
    plt.title("Precision / Recall vs Confidence Threshold")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_precision_recall_vs_threshold.png")
    plt.close()

    # ============================
    # F1 vs Confidence Threshold (IoU=0.50) + best operating point
    #
    # "Accuracy" is intentionally not reported: with a single foreground class
    # and no natural negative-class count, it is not a well-defined detection
    # metric here. F1 answers the same practical question (which confidence
    # threshold to use at inference) without inventing a number.
    # ============================

    s, p, r = precision_recall_vs_score(IOU_50)

    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)

    best = int(np.argmax(f1))

    plt.figure(figsize=(7, 5))
    plt.plot(s, f1, color="tab:green")
    plt.axvline(s[best], linestyle="--", color="gray")
    plt.scatter([s[best]], [f1[best]], color="red", zorder=5, label=f"best F1={f1[best]:.2f} @ score={s[best]:.2f}")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1")
    plt.title("F1 vs Confidence Threshold (IoU=0.50)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_f1_vs_threshold.png")
    plt.close()

    return {
        "best_f1_score_threshold": float(s[best]),
        "best_f1": float(f1[best]),
        "precision_at_best_f1": float(p[best]),
        "recall_at_best_f1": float(r[best]),
        "AP_at_iou_0.50": float(ap_per_iou[IOU_50]),
        "AP_at_iou_0.75": float(ap_per_iou[IOU_75]),
        "AR_at_iou_0.50": float(ar_per_iou[IOU_50]),
        "AR_at_iou_0.75": float(ar_per_iou[IOU_75]),
    }
