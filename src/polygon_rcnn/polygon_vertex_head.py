"""ROI polygon heads with legacy-relative, centered-relative, and structured outputs."""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.modeling.roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY


REPRESENTATIONS = {"legacy_relative", "center_relative", "structured"}
LOSSES = {"vertex", "vertex_iou", "vertex_iou_area"}


def _box_scale(boxes):
    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    return width, height, torch.sqrt(width.square() + height.square()).clamp(min=1.0)


def encode_structured_vertices(vertices, boxes):
    """Encode a quadrilateral as center, double-angle, length, and two end widths."""
    width, height, diagonal = _box_scale(boxes)
    center = vertices.mean(dim=1)
    centered = vertices - center[:, None, :]
    covariance = torch.einsum("nvi,nvj->nij", centered, centered)
    with torch.no_grad():
        _, eigvecs = torch.linalg.eigh(covariance)
        axis = eigvecs[:, :, -1]
        flip = (axis[:, 0] < 0) | ((axis[:, 0].abs() < 1e-7) & (axis[:, 1] < 0))
        axis[flip] *= -1
        normal = torch.stack([-axis[:, 1], axis[:, 0]], dim=1)
        longitudinal = torch.einsum("nvi,ni->nv", centered, axis)
        transverse = torch.einsum("nvi,ni->nv", centered, normal)
        order = longitudinal.argsort(dim=1)
        low_ids, high_ids = order[:, :2], order[:, 2:]
        low_t = transverse.gather(1, low_ids)
        high_t = transverse.gather(1, high_ids)
        low_width = (low_t.max(dim=1).values - low_t.min(dim=1).values).clamp(min=1.0)
        high_width = (high_t.max(dim=1).values - high_t.min(dim=1).values).clamp(min=1.0)
        length = (longitudinal.max(dim=1).values - longitudinal.min(dim=1).values).clamp(min=1.0)
        theta = torch.atan2(axis[:, 1], axis[:, 0])
        center_x = (center[:, 0] - (boxes[:, 0] + boxes[:, 2]) / 2) / width
        center_y = (center[:, 1] - (boxes[:, 1] + boxes[:, 3]) / 2) / height
        return torch.stack([
            center_x,
            center_y,
            torch.cos(2 * theta),
            torch.sin(2 * theta),
            torch.log(length / diagonal),
            torch.log(low_width / diagonal),
            torch.log(high_width / diagonal),
        ], dim=1)


def decode_structured_vertices(parameters, boxes):
    width, height, diagonal = _box_scale(boxes)
    angle_vector = F.normalize(parameters[:, 2:4], dim=1, eps=1e-6)
    theta = 0.5 * torch.atan2(angle_vector[:, 1], angle_vector[:, 0])
    axis = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    normal = torch.stack([-axis[:, 1], axis[:, 0]], dim=1)
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2 + parameters[:, 0] * width
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2 + parameters[:, 1] * height
    center = torch.stack([center_x, center_y], dim=1)
    length = torch.exp(parameters[:, 4].clamp(-6.0, 2.0)) * diagonal
    low_width = torch.exp(parameters[:, 5].clamp(-7.0, 1.0)) * diagonal
    high_width = torch.exp(parameters[:, 6].clamp(-7.0, 1.0)) * diagonal
    low_center = center - axis * (length / 2)[:, None]
    high_center = center + axis * (length / 2)[:, None]
    low_side = normal * (low_width / 2)[:, None]
    high_side = normal * (high_width / 2)[:, None]
    return torch.stack([
        low_center - low_side,
        high_center - high_side,
        high_center + high_side,
        low_center + low_side,
    ], dim=1)


def _decode_vertices(raw, boxes, representation):
    width, height, _ = _box_scale(boxes)
    if representation == "structured":
        return decode_structured_vertices(raw, boxes)
    relative = raw.view(-1, 4, 2)
    if representation == "legacy_relative":
        x = boxes[:, None, 0] + relative[:, :, 0] * width[:, None]
        y = boxes[:, None, 1] + relative[:, :, 1] * height[:, None]
    else:
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        x = center_x[:, None] + relative[:, :, 0] * width[:, None]
        y = center_y[:, None] + relative[:, :, 1] * height[:, None]
    return torch.stack([x, y], dim=2)


def _to_roi_coordinates(vertices, boxes):
    width, height, _ = _box_scale(boxes)
    return torch.stack([
        (vertices[:, :, 0] - boxes[:, None, 0]) / width[:, None],
        (vertices[:, :, 1] - boxes[:, None, 1]) / height[:, None],
    ], dim=2)


def soft_polygon_iou_loss(pred_vertices, gt_vertices, *, grid_size=32, temperature=0.03):
    """Differentiable soft IoU for convex quadrilaterals in proposal-normalized ROIs.

    Four edge half-planes are rasterized on a fixed ROI grid. This is an explicit
    approximation; it is not COCO's polygon IoU and is intended only as a training
    loss. The evaluation metric remains the project COCO evaluator.
    """
    device, dtype = pred_vertices.device, pred_vertices.dtype
    axis = torch.linspace(-0.5, 1.5, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([xx, yy], dim=-1).view(1, 1, grid_size, grid_size, 2)

    def occupancy(vertices):
        start = vertices
        end = torch.roll(vertices, shifts=-1, dims=1)
        edge = end - start
        rel = points - start[:, :, None, None, :]
        cross = edge[:, :, None, None, 0] * rel[..., 1] - edge[:, :, None, None, 1] * rel[..., 0]
        lengths = torch.linalg.vector_norm(edge, dim=2).clamp(min=1e-5)
        signed_area = (start[:, :, 0] * end[:, :, 1] - start[:, :, 1] * end[:, :, 0]).sum(dim=1)
        sign = torch.where(signed_area >= 0, 1.0, -1.0)[:, None, None, None]
        signed_distance = cross * sign / lengths[:, :, None, None]
        return torch.sigmoid(signed_distance / temperature).prod(dim=1)

    pred = occupancy(pred_vertices)
    target = occupancy(gt_vertices).detach()
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target - pred * target).sum(dim=(1, 2)).clamp(min=1e-6)
    return (1.0 - intersection / union).mean()


def _polygon_area_tensor(vertices):
    shifted = torch.roll(vertices, shifts=-1, dims=1)
    return 0.5 * torch.abs(
        (vertices[:, :, 0] * shifted[:, :, 1] - vertices[:, :, 1] * shifted[:, :, 0]).sum(dim=1)
    )


def polygon_vertex_loss(
    raw_predictions,
    instances,
    *,
    representation="legacy_relative",
    loss_mode="vertex",
    iou_weight=0.1,
    area_weight=0.1,
    iou_grid_size=32,
    iou_temperature=0.03,
):
    targets, roi_targets, boxes_all = [], [], []
    for image_instances in instances:
        if len(image_instances) == 0:
            continue
        boxes = image_instances.proposal_boxes.tensor
        keypoints = image_instances.gt_keypoints.tensor
        vertices = keypoints[:, :, :2]
        width, height, _ = _box_scale(boxes)
        if representation == "legacy_relative":
            target = torch.stack([
                (vertices[:, :, 0] - boxes[:, None, 0]) / width[:, None],
                (vertices[:, :, 1] - boxes[:, None, 1]) / height[:, None],
            ], dim=2).flatten(1)
        elif representation == "center_relative":
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            target = torch.stack([
                (vertices[:, :, 0] - cx[:, None]) / width[:, None],
                (vertices[:, :, 1] - cy[:, None]) / height[:, None],
            ], dim=2).flatten(1)
        else:
            target = encode_structured_vertices(vertices, boxes)
        targets.append(target)
        roi_targets.append(_to_roi_coordinates(vertices, boxes))
        boxes_all.append(boxes)

    if not targets:
        return raw_predictions.sum() * 0
    target = torch.cat(targets, dim=0)
    gt_roi = torch.cat(roi_targets, dim=0)
    boxes = torch.cat(boxes_all, dim=0)
    raw_predictions = raw_predictions.reshape(len(target), -1)
    # For structured heads this loss is over the geometric parameters, including
    # the periodic double-angle vector. It avoids an angle discontinuity at +/-pi.
    # Preserve the historical loss scale: the old implementation summed the two
    # coordinates and divided by the number of labeled vertices (4 per instance).
    base = F.smooth_l1_loss(raw_predictions, target, reduction="sum", beta=0.1) / (len(target) * 4)
    if loss_mode == "vertex":
        return base
    predicted_vertices = _decode_vertices(raw_predictions, boxes, representation)
    pred_roi = _to_roi_coordinates(predicted_vertices, boxes)
    iou_loss = soft_polygon_iou_loss(
        pred_roi, gt_roi, grid_size=iou_grid_size, temperature=iou_temperature
    )
    loss = base + iou_weight * iou_loss
    if loss_mode == "vertex_iou_area":
        pred_area = _polygon_area_tensor(pred_roi)
        gt_area = _polygon_area_tensor(gt_roi).clamp(min=1e-6)
        area_loss = (torch.abs(pred_area - gt_area) / gt_area).mean()
        loss = loss + area_weight * area_loss
    return loss


def polygon_vertex_inference(raw_predictions, pred_instances, representation):
    sizes = [len(item) for item in pred_instances]
    chunks = raw_predictions.split(sizes, dim=0)
    for raw, image_instances in zip(chunks, pred_instances):
        if len(image_instances) == 0:
            image_instances.pred_keypoints = raw.new_zeros((0, 4, 3))
            continue
        vertices = _decode_vertices(raw, image_instances.pred_boxes.tensor, representation)
        scores = image_instances.scores[:, None, None].expand(-1, 4, 1)
        image_instances.pred_keypoints = torch.cat([vertices, scores], dim=2)


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class PolygonVertexHead(nn.Module):
    @configurable
    def __init__(
        self, input_shape, *, num_keypoints, conv_dims, fc_dim, loss_weight=1.0,
        representation="legacy_relative", loss_mode="vertex", iou_weight=0.1,
        area_weight=0.1, iou_grid_size=32, iou_temperature=0.03,
    ):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown polygon representation: {representation}")
        if loss_mode not in LOSSES:
            raise ValueError(f"Unknown polygon loss mode: {loss_mode}")
        self.num_keypoints = num_keypoints
        self.loss_weight = loss_weight
        self.representation = representation
        self.loss_mode = loss_mode
        self.iou_weight = iou_weight
        self.area_weight = area_weight
        self.iou_grid_size = iou_grid_size
        self.iou_temperature = iou_temperature
        in_channels = input_shape.channels
        self.conv_layers = nn.ModuleList()
        for channels in conv_dims:
            self.conv_layers.append(Conv2d(in_channels, channels, 3, stride=1, padding=1))
            in_channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, fc_dim)
        output_dim = 7 if representation == "structured" else num_keypoints * 2
        self.predictor = nn.Linear(fc_dim, output_dim)
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(conv.bias, 0)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0.0)
        if representation == "legacy_relative":
            nn.init.constant_(self.predictor.bias, 0.5)
        elif representation == "structured":
            with torch.no_grad():
                self.predictor.bias[2] = 1.0
                self.predictor.bias[4] = -0.5
                self.predictor.bias[5:7] = -2.5

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
            "conv_dims": cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS,
            "fc_dim": 256,
            "loss_weight": cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,
            "representation": os.environ.get("POLYGON_REPRESENTATION", "legacy_relative"),
            "loss_mode": os.environ.get("POLYGON_LOSS", "vertex"),
            "iou_weight": float(os.environ.get("POLYGON_IOU_WEIGHT", "0.1")),
            "area_weight": float(os.environ.get("POLYGON_AREA_WEIGHT", "0.1")),
            "iou_grid_size": int(os.environ.get("POLYGON_IOU_GRID", "32")),
            "iou_temperature": float(os.environ.get("POLYGON_IOU_TEMPERATURE", "0.03")),
        }

    def layers(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = self.pool(x).flatten(start_dim=1)
        return self.predictor(F.relu(self.fc(x)))

    def forward(self, x, instances):
        raw = self.layers(x)
        if self.training:
            loss = polygon_vertex_loss(
                raw, instances,
                representation=self.representation,
                loss_mode=self.loss_mode,
                iou_weight=self.iou_weight,
                area_weight=self.area_weight,
                iou_grid_size=self.iou_grid_size,
                iou_temperature=self.iou_temperature,
            )
            return {"loss_keypoint": loss * self.loss_weight}
        polygon_vertex_inference(raw, instances, self.representation)
        return instances
