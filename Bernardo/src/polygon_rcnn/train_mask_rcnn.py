from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import sys
from pathlib import Path

from src.polygon_rcnn.register_dataset import register_blines

def main():

    register_blines()

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

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

    cfg.OUTPUT_DIR = "./output_maskrcnn"

    trainer = PolygonTrainer(cfg)

    trainer.resume_or_load(resume=False)

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
    main()