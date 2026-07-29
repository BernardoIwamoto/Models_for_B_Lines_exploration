# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np
import torch

from ultralytics.utils import NOT_MACOS14


ArrayLike = torch.Tensor | np.ndarray


def _as_vertex_view(polygons: ArrayLike) -> ArrayLike:
    """Return polygons as (..., 4, 2) without copying when possible."""
    if polygons.shape[-1] == 8:
        return polygons.reshape(*polygons.shape[:-1], 4, 2)
    if polygons.shape[-2:] == (4, 2):
        return polygons
    raise ValueError(f"Expected polygon shape (..., 8) or (..., 4, 2), got {polygons.shape}.")


def polygon8_to_xyxy(polygons: ArrayLike) -> ArrayLike:
    """Convert quadrilateral vertices to enclosing xyxy boxes.

    Args:
        polygons (torch.Tensor | np.ndarray): Polygons with shape (..., 8) or (..., 4, 2).

    Returns:
        (torch.Tensor | np.ndarray): Enclosing boxes with shape (..., 4) in xyxy format.
    """
    vertices = _as_vertex_view(polygons)
    x = vertices[..., 0]
    y = vertices[..., 1]

    if isinstance(polygons, torch.Tensor):
        return torch.stack((x.amin(-1), y.amin(-1), x.amax(-1), y.amax(-1)), dim=-1)
    return np.stack((x.min(-1), y.min(-1), x.max(-1), y.max(-1)), axis=-1).astype(polygons.dtype, copy=False)


def clip_polygon8(polygons: ArrayLike, shape: tuple[int, int] | tuple[int, int, int]) -> ArrayLike:
    """Clip quadrilateral coordinates to image boundaries in-place.

    Args:
        polygons (torch.Tensor | np.ndarray): Polygons with shape (..., 8) or (..., 4, 2).
        shape (tuple[int, int] | tuple[int, int, int]): Image shape as HW or HWC.

    Returns:
        (torch.Tensor | np.ndarray): The clipped input polygon array/tensor.
    """
    h, w = shape[:2]
    vertices = _as_vertex_view(polygons)

    if isinstance(polygons, torch.Tensor):
        if NOT_MACOS14:
            vertices[..., 0].clamp_(0, w)
            vertices[..., 1].clamp_(0, h)
        else:  # Apple macOS14 MPS bug mirrors clip_boxes/clip_coords handling.
            vertices[..., 0] = vertices[..., 0].clamp(0, w)
            vertices[..., 1] = vertices[..., 1].clamp(0, h)
    else:
        vertices[..., 0] = vertices[..., 0].clip(0, w)
        vertices[..., 1] = vertices[..., 1].clip(0, h)

    return polygons


def scale_polygon8(
    img1_shape: tuple[int, int],
    polygons: ArrayLike,
    img0_shape: tuple[int, int],
    ratio_pad: tuple | None = None,
    padding: bool = True,
) -> ArrayLike:
    """Rescale quadrilateral coordinates from one image shape to another.

    This follows the same letterbox scaling convention as `scale_boxes`, but applies it to all four vertices.

    Args:
        img1_shape (tuple[int, int]): Source image shape as HW.
        polygons (torch.Tensor | np.ndarray): Polygons with shape (..., 8) or (..., 4, 2).
        img0_shape (tuple[int, int]): Target image shape as HW.
        ratio_pad (tuple, optional): Tuple of (ratio, pad) from letterbox preprocessing.
        padding (bool): Whether coordinates include letterbox padding.

    Returns:
        (torch.Tensor | np.ndarray): The rescaled input polygon array/tensor clipped to img0_shape.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    vertices = _as_vertex_view(polygons)
    if padding:
        vertices[..., 0] -= pad_x
        vertices[..., 1] -= pad_y
    vertices /= gain
    return clip_polygon8(polygons, img0_shape)


def normalize_polygon8(polygons: ArrayLike, w: int | float, h: int | float) -> ArrayLike:
    """Normalize pixel-space quadrilateral coordinates by image width and height in-place."""
    vertices = _as_vertex_view(polygons)
    vertices[..., 0] /= w
    vertices[..., 1] /= h
    return polygons


def denormalize_polygon8(polygons: ArrayLike, w: int | float, h: int | float) -> ArrayLike:
    """Convert normalized quadrilateral coordinates to pixel coordinates in-place."""
    vertices = _as_vertex_view(polygons)
    vertices[..., 0] *= w
    vertices[..., 1] *= h
    return polygons


def vertex_rmse(pred: ArrayLike, target: ArrayLike) -> ArrayLike:
    """Compute per-polygon vertex RMSE for ordered quadrilateral coordinates."""
    pred_vertices = _as_vertex_view(pred)
    target_vertices = _as_vertex_view(target)
    if pred_vertices.shape != target_vertices.shape:
        raise ValueError(f"Expected matching polygon shapes, got {pred_vertices.shape} and {target_vertices.shape}.")

    if isinstance(pred, torch.Tensor):
        return (pred_vertices - target_vertices).pow(2).mean(dim=(-2, -1)).sqrt()
    return np.sqrt(np.mean((pred_vertices - target_vertices) ** 2, axis=(-2, -1)))
