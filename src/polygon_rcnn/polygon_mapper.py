"""DatasetMapper that restores geometric vertex roles after spatial augmentation."""

import numpy as np

from detectron2.data import DatasetMapper
from detectron2.structures import Keypoints

from src.polygon_rcnn.dataset import _canonical_vertex_order


class CanonicalPolygonDatasetMapper(DatasetMapper):
    """Re-canonicalize transformed quadrilateral keypoints after mapper transforms."""

    def __call__(self, dataset_dict):
        mapped = super().__call__(dataset_dict)
        instances = mapped.get("instances")
        if instances is None or not instances.has("gt_keypoints") or len(instances) == 0:
            return mapped
        values = instances.gt_keypoints.tensor.clone()
        for index in range(len(values)):
            coordinates = values[index, :, :2].detach().cpu().numpy()
            ordered = _canonical_vertex_order(coordinates)
            permutation = [
                int(np.argmin(np.linalg.norm(coordinates - point, axis=1)))
                for point in ordered
            ]
            values[index] = values[index, permutation]
        instances.gt_keypoints = Keypoints(values)
        return mapped
