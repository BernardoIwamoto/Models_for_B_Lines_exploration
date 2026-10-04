#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNS_DIR:-}" ]]; then
  echo 'Defina RUNS_DIR para um diretório com espaço para checkpoints.' >&2
  exit 2
fi

python -m src.polygon_rcnn.validate_dataset --root data
python -m src.polygon_yolo.detect.convert_to_detect

for seed in 0 1 2; do
  echo "=== Mask R-CNN, seed ${seed} ==="
  EXPERIMENT_ID="mask_bbox_s${seed}" SEED="$seed" python -m src.polygon_rcnn.train_mask_rcnn

  echo "=== Faster R-CNN, seed ${seed} ==="
  EXPERIMENT_ID="faster_bbox_s${seed}" SEED="$seed" python -m src.polygon_rcnn.train_faster_rcnn

  echo "=== Polygon Head histórica, seed ${seed} ==="
  EXPERIMENT_ID="polygon_legacy_s${seed}" SEED="$seed" \
    POLYGON_REPRESENTATION=legacy_relative POLYGON_LOSS=vertex POLYGON_FLIP=none \
    python -m src.polygon_rcnn.train_polygon_head

  echo "=== YOLOv11 bbox, seed ${seed} ==="
  EXPERIMENT_ID="yolo11n_bbox_s${seed}" SEED="$seed" python -m src.polygon_yolo.detect.train_yolo_detect
done

echo "Baselines concluídos. Resultados em: $RUNS_DIR"
