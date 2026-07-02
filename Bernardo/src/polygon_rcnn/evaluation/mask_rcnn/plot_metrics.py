from pathlib import Path
import json
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


OUTPUT = Path(__file__).parent / "outputs"

CSV = OUTPUT / "metrics.csv"

df = pd.read_csv(CSV)

# ============================
# Estatísticas
# ============================

summary = {}

for col in [
    "IoU",
    "Dice",
    "Precision",
    "Recall",
    "F1",
    "score",
]:

    summary[col] = {
        "mean": float(df[col].mean()),
        "std": float(df[col].std()),
        "median": float(df[col].median()),
        "min": float(df[col].min()),
        "max": float(df[col].max()),
    }

with open(OUTPUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# ============================
# Histograma IoU
# ============================

plt.figure(figsize=(6,4))

plt.hist(df["IoU"], bins=20)

plt.xlabel("IoU")
plt.ylabel("Count")
plt.title("IoU Distribution")

plt.tight_layout()
plt.savefig(OUTPUT / "histogram_iou.png")
plt.close()

# ============================
# Histograma Dice
# ============================

plt.figure(figsize=(6,4))

plt.hist(df["Dice"], bins=20)

plt.xlabel("Dice")
plt.ylabel("Count")
plt.title("Dice Distribution")

plt.tight_layout()
plt.savefig(OUTPUT / "histogram_dice.png")
plt.close()

# ============================
# Histograma Score
# ============================

plt.figure(figsize=(6,4))

plt.hist(df["score"], bins=20)

plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution")

plt.tight_layout()
plt.savefig(OUTPUT / "histogram_score.png")
plt.close()

# ============================
# Boxplot
# ============================

plt.figure(figsize=(8,5))

plt.boxplot(
    [
        df["IoU"],
        df["Dice"],
        df["Precision"],
        df["Recall"],
        df["F1"],
    ]
)

plt.xticks(
    [1, 2, 3, 4, 5],
    ["IoU", "Dice", "Precision", "Recall", "F1"],
)

plt.ylabel("Metric")

plt.tight_layout()

plt.savefig(OUTPUT / "boxplot_metrics.png")

plt.close()

# ============================
# Score x IoU
# ============================

plt.figure(figsize=(6,5))

plt.scatter(
    df["score"],
    df["IoU"],
    s=20,
)

plt.xlabel("Confidence Score")

plt.ylabel("IoU")

plt.title("Confidence vs IoU")

plt.tight_layout()

plt.savefig(OUTPUT / "scatter_score_iou.png")

plt.close()

plt.figure(figsize=(7,5))

values = np.sort(df["IoU"].values)

cdf = np.arange(1, len(values)+1) / len(values)

plt.plot(values, cdf, linewidth=2)

plt.xlabel("IoU")
plt.ylabel("Cumulative probability")
plt.title("IoU CDF")

plt.grid(True)

plt.savefig(OUTPUT / "iou_cdf.png")

plt.close()

plt.figure(figsize=(6,6))

plt.scatter(
    df["Recall"],
    df["Precision"],
    alpha=0.7
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision vs Recall")

plt.grid(True)

plt.savefig(OUTPUT / "precision_recall.png")

plt.close()

plt.figure(figsize=(6,6))

plt.scatter(
    df["GtArea"],
    df["PredArea"],
    alpha=0.7
)

m = max(df["GtArea"].max(), df["PredArea"].max())

plt.plot([0,m],[0,m],"r--")

plt.xlabel("Ground Truth Area")

plt.ylabel("Predicted Area")

plt.title("Predicted Area vs Ground Truth")

plt.grid(True)

plt.savefig(OUTPUT/"area_scatter.png")

plt.close()

df["AreaError"] = (
    np.abs(df["PredArea"] - df["GtArea"])
    / df["GtArea"].replace(0, np.nan)
)

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=["AreaError"])

plt.figure(figsize=(8,5))

plt.hist(
    df["AreaError"],
    bins=25
)

plt.xlabel("Relative Area Error")

plt.ylabel("Count")

plt.title("Relative Area Error")

plt.grid(True)

plt.savefig(
    OUTPUT/"area_error.png"
)

plt.close()

print()

print("Correlation Score x IoU")

print(
    df["score"].corr(df["IoU"])
)

print()

print("Correlation Area")

print(
    df["GtArea"].corr(df["PredArea"])
)

print("Finished.")

