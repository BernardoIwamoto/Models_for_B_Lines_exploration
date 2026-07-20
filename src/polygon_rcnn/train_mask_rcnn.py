from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
import detectron2.utils.comm as comm
import numpy as np
import torch

import sys
from pathlib import Path

from src.polygon_rcnn.register_dataset import register_blines


class LossEvalHook(HookBase):
    """Computes loss on a validation set, mirroring DefaultTrainer's training-loss logging.

    Detectron2 only evaluates AP/AR on cfg.DATASETS.TEST out of the box; it never runs a
    loss pass on validation data, so overfitting can't be read off the loss curves alone.
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _get_loss(self, data):

        with torch.no_grad():
            loss_dict = self._model(data)

        loss_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_dict.items()
        }

        return sum(loss_dict.values())

    def _do_loss_eval(self):

        losses = [self._get_loss(inputs) for inputs in self._data_loader]

        mean_loss = float(np.mean(losses))

        self.trainer.storage.put_scalar("validation_loss", mean_loss)

        comm.synchronize()

    def after_step(self):

        next_iter = self.trainer.iter + 1

        is_final = next_iter == self.trainer.max_iter

        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


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