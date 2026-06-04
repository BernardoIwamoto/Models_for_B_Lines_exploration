# Ultralytics Source Strategy

Do not modify the installed package in:

```text
.venv/lib/python3.11/site-packages/ultralytics
```

That path is useful for inspection, but edits there are fragile and hard to version-control.

## Recommended Workflow

Use a local editable Ultralytics source checkout:

```bash
mkdir -p external
git clone https://github.com/ultralytics/ultralytics.git external/ultralytics
python -m pip uninstall -y ultralytics
python -m pip install -e external/ultralytics
```

Then implement Polygon YOLO changes inside:

```text
external/ultralytics/ultralytics/
```

This keeps all research changes local, inspectable, and commit-friendly.

## Why Not Patch Site-Packages?

Site-package edits are easy to lose when dependencies are reinstalled, difficult to diff, and poor for research reproducibility.

## First Coding Milestone

After the editable checkout exists:

1. Add `ultralytics/utils/polygon_ops.py`.
2. Add tests for polygon-to-box conversion.
3. Add polygon label parsing.
4. Add `PolygonDetect`.
5. Add `PolygonDetectionLoss`.
