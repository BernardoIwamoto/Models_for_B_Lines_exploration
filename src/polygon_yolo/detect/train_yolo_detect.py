import os

from ultralytics import YOLO


DATA_YAML = "data/yolo_detect/data.yaml"

MODEL = "weights/yolo11n.pt"

EPOCHS = 31

IMGSZ = 640

OUTPUT_DIR = "output_yolo_detect"

SEED = int(os.environ.get("SEED", 0))

# Seed 0 keeps the "train" run name every evaluate/inference script already
# hardcodes; other seeds get their own name so repeated runs don't overwrite each
# other -- run as `SEED=1 python -m src.polygon_yolo.detect.train_yolo_detect`.
RUN_NAME = "train" if SEED == 0 else f"train_seed{SEED}"


def main():

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        val=True,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
