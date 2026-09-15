import numpy as np

from .geometry import polygon_to_mask
from .segmentation import iou


def _as_vertices(polygon_xy):
    """Accepts either a flat (8,) list/array or an already-shaped (4,2) array."""

    arr = np.asarray(polygon_xy, dtype=float)

    return arr.reshape(-1, 2) if arr.ndim == 1 else arr


def vertex_error(pred_xy, gt_xy):
    """Mean Euclidean distance between corresponding vertices, in pixels.

    Assumes both polygons are already in the same canonical vertex order (see
    dataset.py's _canonical_vertex_order) -- vertex k of `pred_xy` is compared to
    vertex k of `gt_xy`, not the nearest one. Only meaningful for a matched
    prediction/ground-truth pair (see the project's existing IoU-based matching in
    evaluate_predictions.py); this does not itself do any matching.
    """

    pred = _as_vertices(pred_xy)
    gt = _as_vertices(gt_xy)

    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def polygon_area(polygon_xy):
    """Exact polygon area via the shoelace formula -- no rasterization, so no
    pixel-grid rounding error, unlike geometry.area(mask).sum() on a rasterized
    mask. Correct for both convex and simple (non-self-intersecting) concave
    polygons; see is_self_intersecting to check that precondition holds.
    """

    vertices = _as_vertices(polygon_xy)

    x, y = vertices[:, 0], vertices[:, 1]

    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def area_error(pred_xy, gt_xy):
    """Relative area error: |A_pred - A_gt| / A_gt."""

    gt_area = polygon_area(gt_xy)

    if gt_area == 0:
        return float("nan")

    return abs(polygon_area(pred_xy) - gt_area) / gt_area


def polygon_iou(pred_xy, gt_xy, shape):
    """IoU between two polygons, computed by rasterizing both (reusing the same
    polygon_to_mask + iou already used for Mask R-CNN's per-instance analysis) and
    comparing pixel masks -- not an analytic polygon-clipping IoU. Simpler and
    reuses tested code; costs a rasterization pass, negligible at this dataset's
    image sizes and instance counts.
    """

    pred_mask = polygon_to_mask(np.asarray(pred_xy).flatten(), shape)
    gt_mask = polygon_to_mask(np.asarray(gt_xy).flatten(), shape)

    return iou(pred_mask, gt_mask)


def _segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 properly crosses segment p3-p4 (shared endpoints, as
    in adjacent polygon edges, don't count -- only a genuine crossing does).
    """

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)

    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def is_self_intersecting(polygon_xy):
    """True if the 4 vertices, connected in order (v0-v1-v2-v3-v0), form a
    self-crossing ("bowtie") shape rather than a simple quadrilateral. In a
    quadrilateral only the two pairs of opposite edges can cross (adjacent edges
    share an endpoint, which isn't a crossing) -- checking those two pairs is
    suffient, no need for a general n-gon self-intersection test. This is the same
    check used ad hoc during the Polygon Head experiments (heatmap: 6.7-20.4%
    invalid; direct regression: 0%); formalized here so it's reusable and testable
    instead of a one-off script.
    """

    v = _as_vertices(polygon_xy)

    if v.shape[0] != 4:
        raise ValueError(f"Expected 4 vertices, got {v.shape[0]}")

    return _segments_intersect(v[0], v[1], v[2], v[3]) or \
        _segments_intersect(v[1], v[2], v[3], v[0])


def orientation(polygon_xy):
    """Long-axis angle of the 4 vertices, in radians, wrapped to [0, pi).

    B-lines are elongated, so the long axis is a meaningful, well-defined property
    even though the model doesn't predict it directly (that's Experiment 5's
    structured parametrization, not this one). Estimated via PCA on the vertex
    coordinates (eigenvector of the largest eigenvalue of their covariance) rather
    than e.g. "vertex 0 to vertex 2", so it doesn't depend on which canonical
    vertex happens to sit at an extremity for a given rotation.

    Wrapped to [0, pi), not [0, 2*pi): this is a line's orientation, not a
    direction -- 10 degrees and 190 degrees describe the same axis. Comparing two
    orientations with a plain subtraction is still wrong even in this range (e.g.
    179 degrees and 1 degree are 2 degrees apart, not 178) -- use
    angular_difference, not `abs(a - b)`.
    """

    v = _as_vertices(polygon_xy)

    centered = v - v.mean(axis=0)

    cov = centered.T @ centered

    eigvals, eigvecs = np.linalg.eigh(cov)

    principal = eigvecs[:, np.argmax(eigvals)]

    angle = np.arctan2(principal[1], principal[0])

    return float(angle % np.pi)


def angular_difference(a, b, period=np.pi):
    """Smallest difference between two angles that live on a circle of the given
    period (default pi, matching orientation()'s range) -- handles wraparound
    correctly, e.g. angular_difference(0.05, pi - 0.05) is small (~0.1), not
    ~(pi - 0.1) as a naive `abs(a - b)` would give.
    """

    diff = (a - b + period / 2) % period - period / 2

    return float(abs(diff))
