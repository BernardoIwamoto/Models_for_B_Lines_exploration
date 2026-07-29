# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import pytest
import torch

from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances


def test_polygon_dataset_preserves_four_vertices():
    dataset = YOLODataset.__new__(YOLODataset)
    dataset.use_polygon = True
    dataset.use_obb = False

    label = {
        "bboxes": np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32),
        "segments": [np.array([[0.1, 0.2], [0.3, 0.2], [0.4, 0.8], [0.0, 0.7]], dtype=np.float32)],
        "keypoints": None,
        "bbox_format": "xywh",
        "normalized": True,
    }

    updated = dataset.update_labels_info(label)

    assert updated["instances"].segments.shape == (1, 4, 2)
    np.testing.assert_allclose(
        updated["instances"].segments.reshape(1, 8),
        np.array([[0.1, 0.2, 0.3, 0.2, 0.4, 0.8, 0.0, 0.7]], dtype=np.float32),
    )


def test_polygon_dataset_rejects_non_quadrilateral_segments():
    dataset = YOLODataset.__new__(YOLODataset)
    dataset.use_polygon = True
    dataset.use_obb = False

    label = {
        "bboxes": np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float32),
        "segments": [np.zeros((5, 2), dtype=np.float32)],
        "keypoints": None,
        "bbox_format": "xywh",
        "normalized": True,
    }

    with pytest.raises(ValueError, match="exactly 4 xy vertices"):
        dataset.update_labels_info(label)


def test_format_returns_normalized_polygon_tensor():
    formatter = Format(bbox_format="xyxy", normalize=True, return_polygon=True, batch_idx=True)
    labels = {
        "img": np.zeros((100, 200, 3), dtype=np.uint8),
        "cls": np.array([[0]], dtype=np.float32),
        "instances": Instances(
            bboxes=np.array([[20, 10, 100, 90]], dtype=np.float32),
            segments=np.array([[[20, 10], [100, 10], [100, 90], [20, 90]]], dtype=np.float32),
            bbox_format="xyxy",
            normalized=False,
        ),
    }

    formatted = formatter(labels)

    assert formatted["polygons"].shape == (1, 8)
    assert torch.allclose(
        formatted["polygons"],
        torch.tensor([[0.1, 0.1, 0.5, 0.1, 0.5, 0.9, 0.1, 0.9]], dtype=torch.float32),
    )
    assert torch.allclose(
        formatted["bboxes"],
        torch.tensor([[0.1, 0.1, 0.5, 0.9]], dtype=torch.float32),
    )


def test_polygon_collate_concatenates_polygons_and_offsets_batch_idx():
    batch = [
        {
            "img": torch.zeros(3, 8, 8),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
            "polygons": torch.tensor([[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]]),
            "batch_idx": torch.zeros(1),
        },
        {
            "img": torch.ones(3, 8, 8),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.3, 0.3, 0.4, 0.4]]),
            "polygons": torch.tensor([[0.3, 0.3, 0.4, 0.3, 0.4, 0.4, 0.3, 0.4]]),
            "batch_idx": torch.zeros(1),
        },
    ]

    collated = YOLODataset.collate_fn(batch)

    assert collated["img"].shape == (2, 3, 8, 8)
    assert collated["polygons"].shape == (2, 8)
    assert torch.equal(collated["batch_idx"], torch.tensor([0.0, 1.0]))
