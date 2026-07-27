import json
from pathlib import Path

import matplotlib.pyplot as plt


# Same dataset/split, same COCOeval code path for all three (see each approach's
# evaluate_coco.py) -- only the "bbox" task is compared, since Faster R-CNN and
# YOLOv11 here have no mask head. Mask R-CNN's own bbox branch is included too,
# as a sanity check on whether the mask branch changes box localization at all.
APPROACHES = {
    "Mask R-CNN (box branch)": "output_maskrcnn/coco_eval/results.json",
    "Faster R-CNN": "output_faster_rcnn/coco_eval/results.json",
    "YOLOv11": "output_yolo_detect/coco_eval/results.json",
}

OUTPUT_DIR = Path("results_bbox_baseline")


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    table = {}

    for name, path in APPROACHES.items():

        path = Path(path)

        if not path.exists():
            print(f"Skipping {name}: {path} not found.")
            continue

        with open(path) as f:
            results = json.load(f)

        table[name] = results["bbox"]

    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(table, f, indent=4)

    if not table:
        print("No results found yet, nothing to plot.")
        return

    metrics = ["AP", "AP50", "AP75"]

    width = 0.25

    plt.figure(figsize=(8, 5))

    for i, metric in enumerate(metrics):
        values = [table[name][metric] for name in table]
        positions = [x + i * width for x in range(len(table))]
        plt.bar(positions, values, width=width, label=metric)

    plt.xticks(
        [x + width for x in range(len(table))],
        list(table.keys()),
    )
    plt.ylabel("AP (%)")
    plt.title("Bounding-Box Detection Baseline Comparison")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_ap.png")
    plt.close()

    print(json.dumps(table, indent=4))


if __name__ == "__main__":
    main()
