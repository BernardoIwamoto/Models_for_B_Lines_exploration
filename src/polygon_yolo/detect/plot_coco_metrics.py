from pathlib import Path
import json

from src.polygon_rcnn.evaluation.common.coco_threshold_plots import plot_coco_threshold_analysis


COCO_DIR = Path("output_yolo_detect/coco_eval")

RESULTS_JSON = COCO_DIR / "results.json"

OUTPUT = Path(__file__).parent / "outputs"

analysis = plot_coco_threshold_analysis(COCO_DIR, OUTPUT, prefix="coco")

summary_path = OUTPUT / "summary.json"

summary = json.load(open(summary_path)) if summary_path.exists() else {}

summary["COCO"] = json.load(open(RESULTS_JSON))
summary["coco_threshold_analysis"] = analysis

json.dump(summary, open(summary_path, "w"), indent=4)

print("Finished COCO threshold plots.")
