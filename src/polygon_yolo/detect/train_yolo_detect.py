from ultralytics import YOLO


DATA_YAML = "data/yolo_detect/data.yaml"

MODEL = "yolo11n.pt"

# Mask/Faster R-CNN train for MAX_ITER=2000 at IMS_PER_BATCH=4 (~256 train images),
# i.e. ~2000*4/256 ≈ 31 passes over the training set. EPOCHS is set to match that same
# images-seen budget; batch size and optimizer are left at Ultralytics' own defaults
# rather than forced to match Detectron2's, since copying batch=4 into YOLO's recipe
# would likely just hurt its BatchNorm statistics without making the comparison fairer.
EPOCHS = 31

IMGSZ = 640

OUTPUT_DIR = "output_yolo_detect"


def main():

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        project=OUTPUT_DIR,
        name="train",
        exist_ok=True,
        val=True,
    )


if __name__ == "__main__":
    main()
