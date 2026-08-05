import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.modeling.roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY


def polygon_vertex_loss(pred_deltas, instances, normalizer=None):
    """Smooth-L1 loss between predicted and ground-truth vertex offsets.

    pred_deltas: (N, K, 2) predicted (tx, ty) per vertex, normalized to each
        instance's own proposal box -- the same proposal-relative, scale-invariant
        encoding Detectron2's own box regression already uses, so the head works the
        same way whether a B-line's box is small or spans most of the image.
    instances: foreground-only Instances (already filtered by ROIHeads before this is
        called), each with `proposal_boxes` and `gt_keypoints` -- the latter already
        canonically ordered by dataset.py, so vertex slot k means the same geometric
        corner for every instance.
    """

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

    # beta=0.1: targets live in a normalized ~[0,1] (box-relative) range, so errors
    # below 10% of the box size are treated quadratically (stable gradient near
    # convergence) and larger ones linearly (robust early in training, when
    # predictions can be far off) -- proportioned to this target scale, unlike
    # Detectron2's box-delta beta defaults which assume a different encoding.
    loss = F.smooth_l1_loss(pred_deltas[valid], targets[valid], reduction="sum", beta=0.1)

    if normalizer is None:
        normalizer = valid.sum().item()

    return loss / normalizer


def polygon_vertex_inference(pred_deltas, pred_instances):
    """Decodes predicted (tx, ty) offsets into absolute image coordinates.

    Uses each instance's final `pred_boxes` (post box-head refinement), mirroring
    how Detectron2's stock keypoint head also decodes against pred_boxes at
    inference despite training against proposal_boxes -- the box only becomes final
    after its own regression head runs, and pooling/vertex regression happen before
    that, at the proposal stage.
    """

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

        # Reuse the instance's own detection score as a per-vertex placeholder: a
        # direct-regression head has no separate per-vertex confidence the way a
        # heatmap's argmax value gives one, and nothing downstream (evaluate_coco.py,
        # inference.py) reads this third column for anything but display anyway.
        score = instances_per_image.scores[:, None].expand(-1, deltas_per_image.shape[1])

        instances_per_image.pred_keypoints = torch.stack([x, y, score], dim=-1)


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class PolygonVertexHead(nn.Module):
    """Direct vertex-coordinate regression head -- the Polygon Head, Phase 3.

    Detectron2's stock keypoint head (KRCNNConvDeconvUpsampleHead) treats each
    vertex as a heatmap classification problem. Phase 2 measured this directly:
    segm_polygon AP=15.0 at the default heatmap resolution, and *worse* (AP=3.8,
    more self-intersecting predictions) after doubling it -- evidence the ceiling
    here isn't spatial resolution, but that heatmap classification gives no reason
    for the 4 independently-classified points to land as a coherent, non-crossing
    quadrilateral, and needs more data/iterations than this dataset provides to
    resolve a fine spatial grid from a head with no pretrained prior for this task.

    This head instead regresses each vertex's (x, y) directly via smooth-L1
    (polygon_vertex_loss), against proposal-box-relative targets -- exactly how
    Detectron2's own box head regresses box deltas. A continuous loss surface
    that rewards getting closer everywhere, not just landing in the right bin.

    Lighter than the heatmap head's conv stack on purpose: there is no upsampling
    path here, so depth is spent before global pooling rather than preserving a
    14x14 spatial map end to end.
    """

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
        # Start every vertex prediction at the proposal box's center -- a neutral
        # zero-th-order guess, closer to right than the (0,0) corner a zero bias
        # would default to.
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
