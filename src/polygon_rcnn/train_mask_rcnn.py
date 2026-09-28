from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pathlib import Path
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.env import seed_all_rng
import torch

import os
import sys

from src.polygon_rcnn.register_dataset import register_blines
from src.polygon_rcnn.evaluation.common.hooks import LossEvalHook
from src.polygon_rcnn.experiment_registry import prepare_run, save_detectron_config, timed_stage


# cfg.SEED alone does nothing -- Detectron2 only reads it inside default_setup(),
# which none of these scripts call (it wants a full argparse.Namespace we don't
# have). seed_all_rng() is what default_setup() itself calls, so this is the same
# effect without pulling in the rest of that machinery.
SEED = int(os.environ.get("SEED", 0))

def main():

    register_blines()

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    cfg.SEED = SEED
    seed_all_rng(SEED)

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TRAIN = ("blines_train",)
    cfg.DATASETS.TEST = ("blines_val",)

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.SOLVER.BASE_LR = 0.00025

    cfg.SOLVER.MAX_ITER = 2000

    cfg.SOLVER.STEPS = []

    cfg.TEST.EVAL_PERIOD = 100

    # Default (5000) exceeds MAX_ITER, so only model_final.pth would ever get saved --
    # validation AP already showed the best checkpoint isn't the last one (peaked at
    # iter 1699, dropped by ~6.5 AP by iter 2000), so we need the intermediate ones to
    # pick from (see evaluation/common/select_best_checkpoint.py).
    cfg.SOLVER.CHECKPOINT_PERIOD = 100

    cfg.OUTPUT_DIR = prepare_run(
        "mask_rcnn_instance_segmentation", SEED,
        {
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "learning_rate": cfg.SOLVER.BASE_LR,
            "max_iter": cfg.SOLVER.MAX_ITER,
        },
        "segm/AP",
    )
    save_detectron_config(cfg.OUTPUT_DIR, cfg)

    resume = False

    trainer = PolygonTrainer(cfg)

    trainer.resume_or_load(resume=resume)

    with timed_stage(cfg.OUTPUT_DIR, "training_wall_seconds"):
        trainer.train()


if __name__ == "__main__":
    class PolygonTrainer(DefaultTrainer):

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):

            if output_folder is None:
                output_folder = f"{cfg.OUTPUT_DIR}/inference"

            return COCOEvaluator(
                dataset_name,
                output_dir=output_folder,
            )

        def build_hooks(self):

            hooks = super().build_hooks()

            val_loader = build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, is_train=False),
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
