
import numpy as np
import pytest
import torch

from ultralytics.utils.polygon_ops import (
    clip_polygon8,
    denormalize_polygon8,
    normalize_polygon8,
    polygon8_to_xyxy,
    scale_polygon8,
    vertex_rmse,
)


def test_polygon8_to_xyxy_numpy_flat():
    polygons = np.array(
        [
            [10, 20, 30, 15, 25, 60, 5, 55],
            [0, 0, 8, 0, 8, 4, 0, 4],
        ],
        dtype=np.float32,
    )

    boxes = polygon8_to_xyxy(polygons)

    np.testing.assert_allclose(boxes, np.array([[5, 15, 30, 60], [0, 0, 8, 4]], dtype=np.float32))
    assert boxes.dtype == polygons.dtype


def test_polygon8_to_xyxy_torch_vertex_view():
    polygons = torch.tensor(
        [
            [[10, 20], [30, 15], [25, 60], [5, 55]],
            [[0, 0], [8, 0], [8, 4], [0, 4]],
        ],
        dtype=torch.float32,
    )

    boxes = polygon8_to_xyxy(polygons)

    assert torch.allclose(boxes, torch.tensor([[5, 15, 30, 60], [0, 0, 8, 4]], dtype=torch.float32))


def test_clip_polygon8_clips_numpy_in_place():
    polygons = np.array([[-5, 2, 12, -3, 20, 9, 4, 15]], dtype=np.float32)

    returned = clip_polygon8(polygons, (10, 12))

    assert returned is polygons
    np.testing.assert_allclose(polygons, np.array([[0, 2, 12, 0, 12, 9, 4, 10]], dtype=np.float32))


def test_clip_polygon8_clips_torch_in_place():
    polygons = torch.tensor([[-5, 2, 12, -3, 20, 9, 4, 15]], dtype=torch.float32)

    returned = clip_polygon8(polygons, (10, 12))

    assert returned is polygons
    assert torch.allclose(polygons, torch.tensor([[0, 2, 12, 0, 12, 9, 4, 10]], dtype=torch.float32))


def test_scale_polygon8_removes_letterbox_padding_and_clips():
    polygons = torch.tensor([[20, 40, 60, 40, 60, 80, 20, 80]], dtype=torch.float32)

    scaled = scale_polygon8((100, 100), polygons, (50, 100), ratio_pad=((1.0, 1.0), (0, 25)))

    assert scaled is polygons
    assert torch.allclose(polygons, torch.tensor([[20, 15, 60, 15, 60, 50, 20, 50]], dtype=torch.float32))


def test_normalize_and_denormalize_polygon8_round_trip():
    polygons = np.array([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=np.float32)

    normalized = normalize_polygon8(polygons.copy(), w=100, h=200)
    restored = denormalize_polygon8(normalized, w=100, h=200)

    np.testing.assert_allclose(restored, polygons)


def test_vertex_rmse_matches_ordered_vertices():
    pred = torch.tensor([[0, 0, 2, 0, 2, 2, 0, 2]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 4, 0, 4, 4, 0, 4]], dtype=torch.float32)

    rmse = vertex_rmse(pred, target)

    assert torch.allclose(rmse, torch.tensor([torch.sqrt(torch.tensor(2.0))]))


def test_polygon_ops_reject_invalid_shape():
    with pytest.raises(ValueError, match="Expected polygon shape"):
        polygon8_to_xyxy(np.zeros((2, 6), dtype=np.float32))
