from pathlib import Path

from src.polygon_rcnn.evaluation.common.io import load_metrics
from src.polygon_rcnn.evaluation.common.utils import get_series
from src.polygon_rcnn.evaluation.common.plots import plot_curves


METRICS = "output_maskrcnn/metrics.json"

OUTPUT = Path(__file__).parent / "outputs"

metrics = load_metrics(METRICS)

curves = [
    (*get_series(metrics, "total_loss"), "Total"),
    (*get_series(metrics, "loss_cls"), "Classification"),
    (*get_series(metrics, "loss_box_reg"), "Box"),
    (*get_series(metrics, "loss_mask"), "Mask"),
]

plot_curves(
    curves,
    "Training Loss",
    "Loss",
    OUTPUT / "losses.png",
)

curves = [
    (*get_series(metrics, "loss_rpn_cls"), "RPN cls"),
    (*get_series(metrics, "loss_rpn_loc"), "RPN loc"),
]

plot_curves(
    curves,
    "RPN Loss",
    "Loss",
    OUTPUT / "rpn_losses.png",
)

curves = [
    (*get_series(metrics, "lr"), "Learning Rate"),
]

plot_curves(
    curves,
    "Learning Rate",
    "LR",
    OUTPUT / "learning_rate.png",
)

curves = [
    (*get_series(metrics, "fast_rcnn/cls_accuracy"), "Classification"),
    (*get_series(metrics, "fast_rcnn/fg_cls_accuracy"), "Foreground"),
    (*get_series(metrics, "mask_rcnn/accuracy"), "Mask"),
]

plot_curves(
    curves,
    "Accuracy",
    "Accuracy",
    OUTPUT / "accuracy.png",
)

curves = [
    (*get_series(metrics, "segm/AP"), "Segmentation AP"),
    (*get_series(metrics, "segm/AP50"), "Segmentation AP50"),
]

plot_curves(
    curves,
    "Validation AP (segm) during training",
    "AP",
    OUTPUT / "val_segm_ap.png",
)

curves = [
    (*get_series(metrics, "bbox/AP"), "BBox AP"),
    (*get_series(metrics, "bbox/AP50"), "BBox AP50"),
]

plot_curves(
    curves,
    "Validation AP (bbox) during training",
    "AP",
    OUTPUT / "val_bbox_ap.png",
)

curves = [
    (*get_series(metrics, "total_loss"), "Training Loss"),
    (*get_series(metrics, "validation_loss"), "Validation Loss"),
]

plot_curves(
    curves,
    "Training vs Validation Loss",
    "Loss",
    OUTPUT / "train_vs_val_loss.png",
)