from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

from src.polygon_rcnn.dataset import yolo_polygon_to_detectron


DATA_ROOT = Path("data")

# Matches dataset.py's _canonical_vertex_order: index 0 is always the topmost
# vertex, followed by a consistent winding order -- a geometric role, not a
# left/right body part, so there is nothing to swap under a horizontal flip.
KEYPOINT_NAMES = ["v0", "v1", "v2", "v3"]

KEYPOINT_FLIP_MAP = []


def register_blines():

    DatasetCatalog.register(
        "blines_train",
        lambda: yolo_polygon_to_detectron(
            DATA_ROOT / "train" / "images",
            DATA_ROOT / "train" / "labels",
        ),
    )

    MetadataCatalog.get("blines_train").thing_classes = ["bline"]
    MetadataCatalog.get("blines_train").evaluator_type = "coco"
    MetadataCatalog.get("blines_train").keypoint_names = KEYPOINT_NAMES
    MetadataCatalog.get("blines_train").keypoint_flip_map = KEYPOINT_FLIP_MAP

    DatasetCatalog.register(
        "blines_val",
        lambda: yolo_polygon_to_detectron(
            DATA_ROOT / "val" / "images",
            DATA_ROOT / "val" / "labels",
        ),
    )

    MetadataCatalog.get("blines_val").thing_classes = ["bline"]
    MetadataCatalog.get("blines_val").evaluator_type = "coco"
    MetadataCatalog.get("blines_val").keypoint_names = KEYPOINT_NAMES
    MetadataCatalog.get("blines_val").keypoint_flip_map = KEYPOINT_FLIP_MAP