# B-Lines Polygon Detection

Research workspace for object detection experiments on ultrasound B-Line annotations represented as quadrilaterals:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

The first implementation target is Polygon YOLO11: a YOLO-style single-stage detector that directly regresses 8 polygon vertex coordinates instead of standard boxes or OBB parameters.

## Environment

Use Python 3.10 or 3.11. The project deliberately avoids Python 3.13 because PyTorch and Detectron2 compatibility is less reliable there.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Optional local runtime configuration:

```bash
export YOLO_CONFIG_DIR=.ultralytics
```

Detectron2 is handled separately. See `requirements-detectron2.txt`.

## Planned Implementation Order

1. Install or vendor the Ultralytics source code so modifications are local and reproducible.
2. Add polygon geometry helpers.
3. Add polygon dataset parsing for `class + 8 coordinates`.
4. Add `PolygonDetect` head.
5. Add `PolygonDetectionLoss` using SmoothL1 vertex regression.
6. Add validation support using derived enclosing boxes first.
7. Add later metrics: Polygon IoU and Vertex RMSE.

## Representation

Primary target:

```text
x1 y1 x2 y2 x3 y3 x4 y4
```

Compatibility geometry:

```text
xyxy = enclosing_box(vertices)
```

The enclosing box is only for assignment, NMS, and traditional mAP compatibility. It is not the primary learned representation.
