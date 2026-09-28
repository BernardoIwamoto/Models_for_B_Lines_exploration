import os
from pathlib import Path

from ultralytics import YOLO

from src.polygon_rcnn.experiment_registry import prepare_run, timed_stage


DATA_YAML = "data/yolo_detect/data.yaml"

MODEL = "weights/yolo11n.pt"

EPOCHS = 31

IMGSZ = 640

SEED = int(os.environ.get("SEED", 0))


def main():

    run_dir = Path(prepare_run(
        "yolo11n_bbox", SEED,
        {
            "epochs": EPOCHS,
            "batch_size": int(os.environ.get("BATCH_SIZE", 16)),
            "learning_rate": "Ultralytics default",
            "image_size": IMGSZ,
            "weights": MODEL,
        },
        "Ultralytics best.pt fitness (bbox mAP50-95)",
    ))
    model = YOLO(MODEL)

    with timed_stage(run_dir, "training_wall_seconds"):
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            val=True,
            seed=SEED,
            batch=int(os.environ.get("BATCH_SIZE", 16)),
        )


if __name__ == "__main__":
    main()
