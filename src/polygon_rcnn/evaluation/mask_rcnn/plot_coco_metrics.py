from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


COCO_DIR = Path("output_maskrcnn/coco_eval")

OUTPUT = Path(__file__).parent / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

# precision: [T, R, K, A, M]  recall: [T, K, A, M]  scores: [T, R, K, A, M]
precision = np.load(COCO_DIR / "precision.npy")
recall = np.load(COCO_DIR / "recall.npy")
scores = np.load(COCO_DIR / "scores.npy")
iou_thresholds = np.load(COCO_DIR / "iou_thresholds.npy")
recall_thresholds = np.load(COCO_DIR / "recall_thresholds.npy")

CAT = 0
AREA = 0
MAX_DETS = -1

IOU_50 = int(np.argmin(np.abs(iou_thresholds - 0.5)))
IOU_75 = int(np.argmin(np.abs(iou_thresholds - 0.75)))

# ============================
# PR curves per IoU threshold (official COCOeval data)
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
plt.savefig(OUTPUT / "coco_pr_curves.png")
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
plt.savefig(OUTPUT / "coco_precision_vs_iou.png")
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
plt.savefig(OUTPUT / "coco_recall_vs_iou.png")
plt.close()

# ============================
# Precision / Recall / F1 vs Confidence Threshold
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
plt.savefig(OUTPUT / "coco_precision_recall_vs_threshold.png")
plt.close()

# ============================
# F1 vs Confidence Threshold (IoU=0.50) + best operating point
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
plt.savefig(OUTPUT / "coco_f1_vs_threshold.png")
plt.close()

# ============================
# Merge summary numbers into summary.json
# ============================

summary_path = OUTPUT / "summary.json"

summary = json.load(open(summary_path)) if summary_path.exists() else {}

summary["coco_threshold_analysis"] = {
    "best_f1_score_threshold": float(s[best]),
    "best_f1": float(f1[best]),
    "precision_at_best_f1": float(p[best]),
    "recall_at_best_f1": float(r[best]),
    "AP_at_iou_0.50": float(ap_per_iou[IOU_50]),
    "AP_at_iou_0.75": float(ap_per_iou[IOU_75]),
    "AR_at_iou_0.50": float(ar_per_iou[IOU_50]),
    "AR_at_iou_0.75": float(ar_per_iou[IOU_75]),
}

json.dump(summary, open(summary_path, "w"), indent=4)

print("Finished COCO threshold plots.")
