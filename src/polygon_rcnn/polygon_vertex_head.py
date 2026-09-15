import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.modeling.roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY


def polygon_vertex_loss(pred_deltas, instances, normalizer=None):

    targets = []
    valid = []

    for instances_per_image in instances:

        if len(instances_per_image) == 0:
            continue

        boxes = instances_per_image.proposal_boxes.tensor
        keypoints = instances_per_image.gt_keypoints.tensor

        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)

        tx = (keypoints[:, :, 0] - boxes[:, 0:1]) / widths[:, None]
        ty = (keypoints[:, :, 1] - boxes[:, 1:2]) / heights[:, None]

        targets.append(torch.stack([tx, ty], dim=-1))
        valid.append(keypoints[:, :, 2] > 0)

    if len(targets) == 0:
        return pred_deltas.sum() * 0

    targets = torch.cat(targets, dim=0)
    valid = torch.cat(valid, dim=0)

    if valid.sum() == 0:
        return pred_deltas.sum() * 0

    loss = F.smooth_l1_loss(pred_deltas[valid], targets[valid], reduction="sum", beta=0.1)

    if normalizer is None:
        normalizer = valid.sum().item()

    return loss / normalizer


def polygon_vertex_inference(pred_deltas, pred_instances):

    num_instances_per_image = [len(i) for i in pred_instances]
    pred_deltas = pred_deltas.split(num_instances_per_image, dim=0)

    for deltas_per_image, instances_per_image in zip(pred_deltas, pred_instances):

        if len(instances_per_image) == 0:
            instances_per_image.pred_keypoints = deltas_per_image.new_zeros(
                (0, deltas_per_image.shape[1], 3)
            )
            continue

        boxes = instances_per_image.pred_boxes.tensor

        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)

        x = boxes[:, 0:1] + deltas_per_image[:, :, 0] * widths[:, None]
        y = boxes[:, 1:2] + deltas_per_image[:, :, 1] * heights[:, None]

        score = instances_per_image.scores[:, None].expand(-1, deltas_per_image.shape[1])

        instances_per_image.pred_keypoints = torch.stack([x, y, score], dim=-1)


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class PolygonVertexHead(nn.Module):

    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, fc_dim, loss_weight=1.0):

        super().__init__()

        self.num_keypoints = num_keypoints
        self.loss_weight = loss_weight

        in_channels = input_shape.channels

        self.conv_layers = nn.ModuleList()

        for layer_channels in conv_dims:
            self.conv_layers.append(Conv2d(in_channels, layer_channels, 3, stride=1, padding=1))
            in_channels = layer_channels

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(in_channels, fc_dim)

        self.predictor = nn.Linear(fc_dim, num_keypoints * 2)

        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(conv.bias, 0)

        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0.5)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
            "conv_dims": cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS,
            "fc_dim": 256,
            "loss_weight": cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,
        }

    def layers(self, x):

        for conv in self.conv_layers:
            x = F.relu(conv(x))

        x = self.pool(x).flatten(start_dim=1)

        x = F.relu(self.fc(x))

        x = self.predictor(x)

        return x.view(-1, self.num_keypoints, 2)

    def forward(self, x, instances):

        deltas = self.layers(x)

        if self.training:
            return {"loss_keypoint": polygon_vertex_loss(deltas, instances) * self.loss_weight}

        polygon_vertex_inference(deltas, instances)

        return instances
