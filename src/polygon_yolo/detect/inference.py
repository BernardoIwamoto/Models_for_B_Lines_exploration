from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detectron2.data import DatasetCatalog

from src.polygon_rcnn.register_dataset import register_blines


# Same path evaluate_coco.py uses -- Ultralytics saved this run under runs/detect/
# instead of directly under OUTPUT_DIR from train_yolo_detect.py.
MODEL_PATH = "runs/detect/output_yolo_detect/train/weights/best.pt"

TEST_DATASET = "blines_val"

OUTPUT_DIR = "results_yolo_detect"

SCORE_THRESHOLD = 0.50


def main():

    register_blines()

    model = YOLO(MODEL_PATH)

    # Same blines_val dataset dicts as the Detectron2 pipelines, so ground truth is
    # drawn identically across all three models' visualizations.
    dataset = DatasetCatalog.get(TEST_DATASET)

    output = Path(OUTPUT_DIR)
    output.mkdir(exist_ok=True)

    for sample in dataset:

        image = cv2.imread(sample["file_name"])

        vis = image.copy()

        # Ground Truth (verde)

        for ann in sample["annotations"]:

            polygon = np.array(
                ann["segmentation"][0],
                dtype=np.int32,
            ).reshape(-1, 2)

            cv2.polylines(
                vis,
                [polygon],
                True,
                (0, 255, 0),
                2,
            )

        # Predições -- bounding box (vermelho)

        result = model.predict(
            sample["file_name"],
            conf=SCORE_THRESHOLD,
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()

        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                vis,
                f"{score:.3f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        out_name = Path(sample["file_name"]).name

        cv2.imwrite(
            str(output / out_name),
            vis,
        )

        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
