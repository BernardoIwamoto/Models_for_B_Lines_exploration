import numpy as np

from detectron2.data.datasets.coco import convert_to_coco_dict
from detectron2.evaluation import DatasetEvaluator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_polygon_coco_gt(dataset_name):
    """Builds a pycocotools COCO ground-truth object directly from the registered
    dataset dicts, in memory -- no json file needed.

    Reuses Detectron2's own convert_to_coco_dict (the same function COCOEvaluator
    calls internally when there's no cached json), so this is guaranteed to use the
    identical image_id/category_id scheme as every other evaluator in this project.
    """

    coco_gt = COCO()
    coco_gt.dataset = convert_to_coco_dict(dataset_name)
    coco_gt.createIndex()

    return coco_gt


def polygon_to_coco_prediction(image_id, polygon_xy, score, category_id=0):
    """Builds one COCO-format "segm" prediction entry from a flat (x1,y1,...,x4,y4)
    polygon. Includes bbox (the polygon's own tight extent) because
    pycocotools.coco.COCO.loadRes requires it to be present to take the code path
    that preserves our polygon segmentation instead of erroring on it as if it
    were RLE (see evaluate_polygon_segm's docstring).
    """

    xs = polygon_xy[0::2]
    ys = polygon_xy[1::2]

    x0, y0 = float(min(xs)), float(min(ys))
    w, h = float(max(xs)) - x0, float(max(ys)) - y0

    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x0, y0, w, h],
        "segmentation": [list(map(float, polygon_xy))],
        "score": float(score),
    }


def evaluate_polygon_segm(coco_gt, predictions):
    """Runs pycocotools COCOeval (iouType="segm") on polygon-format predictions
    against a COCO ground-truth object. `predictions` is a list of dicts with
    image_id, category_id, bbox, segmentation (list-of-floats polygon, COCO's
    native polygon format, not RLE) and score -- the exact format GT segmentations
    in this project already use, so no rasterization/encoding step is needed.

    `bbox` must be present (pycocotools.coco.COCO.loadRes branches on it before it
    even looks at `segmentation`; without it, this version tries to treat
    `segmentation` as RLE and crashes). Verified in isolation that its *value*
    doesn't affect the segm IoU pycocotools actually computes -- loadRes only uses
    it to fill in `area` and only synthesizes a segmentation from it when one isn't
    already present, which is never the case here -- so any consistent box works;
    build_polygon_coco_predictions below uses the polygon's own tight bounding box.

    Returns (results_dict, coco_eval) -- the raw COCOeval object is returned too so
    callers that also want the full precision/recall/scores arrays (for threshold
    plots) don't need to re-run evaluation a second time.
    """

    if len(predictions) == 0:
        empty = {k: float("nan") for k in ("AP", "AP50", "AP75", "APs", "APm", "APl")}
        return {"segm_polygon": empty}, None

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "AP": float(coco_eval.stats[0]) * 100,
        "AP50": float(coco_eval.stats[1]) * 100,
        "AP75": float(coco_eval.stats[2]) * 100,
        "APs": float(coco_eval.stats[3]) * 100,
        "APm": float(coco_eval.stats[4]) * 100,
        "APl": float(coco_eval.stats[5]) * 100,
    }

    return {"segm_polygon": results}, coco_eval


class PolygonSegmEvaluator(DatasetEvaluator):
    """Converts each predicted instance's keypoints (the Polygon Head's 4 vertices)
    straight into a polygon "segmentation" entry and evaluates with the same
    segm-task COCOeval used for Mask R-CNN's own segm AP -- directly comparable
    numbers, no metric reimplementation.

    Meant to run *alongside* COCOEvaluator (see train_polygon_head.py's
    build_evaluator, which composes the two via DatasetEvaluators), including
    during the periodic mid-training eval -- so select_best_checkpoint.py can pick
    checkpoints by "segm_polygon/AP", the metric the task actually cares about,
    instead of the "bbox/AP" proxy the report describes as a known limitation.
    The extra cost is small: the network's forward pass over the validation set
    already happens for the bbox/keypoints eval this runs alongside; this only adds
    CPU-side polygon conversion and a COCOeval pass on the same predictions.
    """

    def __init__(self, dataset_name):

        self._dataset_name = dataset_name
        self._coco_gt = None
        self.coco_eval = None  # last COCOeval object, for callers that want the
        # raw precision/recall/scores arrays without re-running evaluation.

    def reset(self):

        self._predictions = []

        if self._coco_gt is None:
            self._coco_gt = build_polygon_coco_gt(self._dataset_name)

    def process(self, inputs, outputs):

        for inp, out in zip(inputs, outputs):

            instances = out["instances"].to("cpu")

            if not instances.has("pred_keypoints") or len(instances) == 0:
                continue

            keypoints = instances.pred_keypoints.numpy()
            scores = instances.scores.numpy()

            for kp, score in zip(keypoints, scores):

                self._predictions.append(
                    polygon_to_coco_prediction(
                        inp["image_id"],
                        kp[:, :2].flatten(),
                        score,
                    )
                )

    def evaluate(self):

        results, coco_eval = evaluate_polygon_segm(self._coco_gt, self._predictions)

        self.coco_eval = coco_eval

        return results
