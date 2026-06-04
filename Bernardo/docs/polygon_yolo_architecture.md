# Polygon YOLO11 Architecture Plan

## Current YOLO11 Detection Path

```text
yolo11.yaml
  -> Detect(P3, P4, P5)
    -> regression branch: 4 * reg_max DFL box distances
    -> classification branch: nc logits
    -> bbox decode: dist2bbox(anchor-relative distances)
    -> inference output: xyxy boxes + class scores
    -> NMS/metrics/results expect 4 box coordinates
```

The training loss uses `v8DetectionLoss`:

```text
model outputs
  -> parse_output()
  -> make_anchors()
  -> preprocess labels from xywh to xyxy
  -> bbox_decode()
  -> TaskAlignedAssigner
  -> BboxLoss + BCE classification + DFL
```

## Polygon YOLO11 Target Path

```text
yolo11-polygon.yaml
  -> PolygonDetect(P3, P4, P5)
    -> vertex branch: 8 direct coordinates
    -> classification branch: nc logits
    -> derived boxes: polygon8_to_xyxy(vertices)
    -> inference output: polygon vertices + class scores
    -> NMS/metrics use derived boxes initially
```

## Initial Loss

Version 1 only:

```text
classification loss
+ SmoothL1(vertices)
```

The code should reserve a clean extension point for:

```text
+ Polygon IoU loss
+ custom geometric loss
```

but those should not be implemented in the first baseline.

## Files Expected To Change Once Coding Begins

`ultralytics/cfg/models/11/yolo11-polygon.yaml`

Defines the experimental model config and uses `PolygonDetect`.

`ultralytics/nn/modules/head.py`

Adds `PolygonDetect`, keeping classification and geometry branches separate.

`ultralytics/utils/loss.py`

Adds `PolygonDetectionLoss`, initially SmoothL1 over assigned foreground vertices.

`ultralytics/nn/tasks.py`

Routes polygon models to `PolygonDetectionLoss`.

`ultralytics/data/dataset.py`

Loads polygon labels and stores both original vertices and derived boxes.

`ultralytics/data/augment.py`

Applies image augmentations to polygon vertices and recomputes derived boxes.

`ultralytics/utils/polygon_ops.py`

New geometry helper module for polygon conversion, clipping, scaling, and metrics.

## Dependency Rule

Box geometry is allowed only as derived compatibility geometry. The learned target remains the ordered 8-coordinate quadrilateral.
