from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

from dataset import yolo_polygon_to_detectron


DATA_ROOT = Path("data")


def register_blines():

    DatasetCatalog.register(
        "blines_train",
        lambda: yolo_polygon_to_detectron(
            DATA_ROOT / "train" / "images",
            DATA_ROOT / "train" / "labels",
        ),
    )

    MetadataCatalog.get("blines_train").thing_classes = ["bline"]

    DatasetCatalog.register(
        "blines_val",
        lambda: yolo_polygon_to_detectron(
            DATA_ROOT / "val" / "images",
            DATA_ROOT / "val" / "labels",
        ),
    )

    MetadataCatalog.get("blines_val").thing_classes = ["bline"]