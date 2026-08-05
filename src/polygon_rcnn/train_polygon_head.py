from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
import torch

from src.polygon_rcnn.register_dataset import register_blines
from src.polygon_rcnn.evaluation.common.hooks import LossEvalHook

# Registers "PolygonVertexHead" into Detectron2's ROI_KEYPOINT_HEAD_REGISTRY as a
# side effect of the @ROI_KEYPOINT_HEAD_REGISTRY.register() decorator -- must be
# imported before cfg.MODEL.ROI_KEYPOINT_HEAD.NAME below is looked up by name.
from src.polygon_rcnn.polygon_vertex_head import PolygonVertexHead  # noqa: F401


NUM_KEYPOINTS = 4

# Phase 2 tried Detectron2's stock heatmap-classification keypoint head at two
# resolutions: default (segm_polygon AP=15.0) and doubled (AP=3.8, worse -- see
# commits 5b676fa, 34e2a06, 938976a). This run instead uses PolygonVertexHead, a
# direct vertex-coordinate regression head with its own smooth-L1 loss (see
# polygon_vertex_head.py for the full reasoning). POOLER_RESOLUTION here now only
# controls the *input* feature resolution fed to that head's conv stack before
# global pooling, not an output heatmap size, so the "coarser vs finer" tradeoff
# that hurt the heatmap head doesn't apply the same way -- kept at 14 to match
# Phase 1/2's protocol and avoid adding a second confound.
KEYPOINT_POOLER_RESOLUTION = 14


def main():

    register_blines()

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TRAIN = ("blines_train",)
    cfg.DATASETS.TEST = ("blines_val",)

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "PolygonVertexHead"

    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = NUM_KEYPOINTS

    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = KEYPOINT_POOLER_RESOLUTION

    # PolygonVertexHead global-average-pools before its FC layers (no upsampling
    # path, unlike the heatmap head), so it doesn't need 8 conv layers preserving a
    # 14x14 map end to end -- that default was tuned for the heatmap head. 4 lighter
    # (256-channel) layers are plenty of depth before pooling for a regression head.
    cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (256, 256, 256, 256)

    # pycocotools' keypoint OKS eval needs one sigma per keypoint. COCOEvaluator
    # auto-runs a "keypoints" task (in addition to bbox) whenever predictions carry
    # pred_keypoints -- including during TEST.EVAL_PERIOD's mid-training eval -- and
    # crashes without this, since COCO's 17 human-joint default sigmas don't match
    # our 4. We don't use this OKS-AP as the reported metric (see
    # evaluation/polygon_head/evaluate_coco.py, which converts vertices back to a
    # polygon and reuses the same segm-task COCOeval as Mask/Faster R-CNN), so the
    # exact values here don't matter -- only that 4 of them exist.
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.05] * NUM_KEYPOINTS

    # Each polygon vertex is canonicalized by its own geometric role (topmost point
    # first, then a consistent winding order -- see dataset.py), which a horizontal
    # flip would silently invalidate: flipping mirrors the winding direction, so
    # vertex slot 1 would mean "next point clockwise" for un-flipped samples and
    # "next point counter-clockwise" for flipped ones, adding label noise Detectron2's
    # stock keypoint-flip handling (built for named left/right body joints, not
    # geometric roles) doesn't fix. Simplest safe fix: turn flip off for this run.
    # This is a deliberate, disclosed protocol difference from Mask/Faster R-CNN,
    # which train with Detectron2's default horizontal flip.
    cfg.INPUT.RANDOM_FLIP = "none"

    # Same optimization budget as train_mask_rcnn.py/train_faster_rcnn.py.
    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.SOLVER.BASE_LR = 0.00025

    cfg.SOLVER.MAX_ITER = 2000

    cfg.SOLVER.STEPS = []

    cfg.TEST.EVAL_PERIOD = 100

    cfg.SOLVER.CHECKPOINT_PERIOD = 100

    cfg.OUTPUT_DIR = "./output_polygon_head"

    resume = False

    # Detectron2's metrics.json is append-only, so a fresh (non-resumed) run must
    # clear it first or its log gets mixed with older runs (see train_mask_rcnn.py).
    if not resume:
        metrics_file = Path(cfg.OUTPUT_DIR) / "metrics.json"
        if metrics_file.exists():
            metrics_file.unlink()

    trainer = PolygonHeadTrainer(cfg)

    trainer.resume_or_load(resume=resume)

    trainer.train()


if __name__ == "__main__":
    class PolygonHeadTrainer(DefaultTrainer):

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):

            if output_folder is None:
                output_folder = f"{cfg.OUTPUT_DIR}/inference"

            return COCOEvaluator(
                dataset_name,
                output_dir=output_folder,
                # kpt_oks_sigmas is only read from cfg.TEST.KEYPOINT_OKS_SIGMAS when a
                # (deprecated) CfgNode is passed as `tasks`; passing it explicitly here
                # is the only way it actually reaches pycocotools. Without it, the
                # periodic mid-training eval crashes on the shape mismatch (COCO's 17
                # default sigmas vs our 4 keypoints).
                kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
            )

        def build_hooks(self):

            hooks = super().build_hooks()

            val_loader = build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, is_train=True),
            )

            hooks.insert(
                -1,
                LossEvalHook(
                    self.cfg.TEST.EVAL_PERIOD,
                    self.model,
                    val_loader,
                ),
            )

            return hooks
    main()
