import torch
import numpy as np
import detectron2.utils.comm as comm
from detectron2.engine.hooks import HookBase


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
