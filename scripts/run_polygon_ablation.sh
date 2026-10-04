#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNS_DIR:-}" ]]; then
  echo 'Defina RUNS_DIR para um diretório com espaço para checkpoints.' >&2
  exit 2
fi

python -m src.polygon_rcnn.validate_dataset --root data

for seed in 0 1 2; do
  echo "=== Relative-to-center vertices, seed ${seed} ==="
  EXPERIMENT_ID="polygon_center_s${seed}" SEED="$seed" \
    POLYGON_REPRESENTATION=center_relative POLYGON_LOSS=vertex POLYGON_FLIP=none \
    python -m src.polygon_rcnn.train_polygon_head

  echo "=== Structured geometry + vertex loss, seed ${seed} ==="
  EXPERIMENT_ID="polygon_struct_s${seed}" SEED="$seed" \
    POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex POLYGON_FLIP=none \
    python -m src.polygon_rcnn.train_polygon_head

  echo "=== Structured geometry + polygon IoU, seed ${seed} ==="
  EXPERIMENT_ID="polygon_struct_iou_s${seed}" SEED="$seed" \
    POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex_iou \
    POLYGON_IOU_WEIGHT=0.1 POLYGON_FLIP=none \
    python -m src.polygon_rcnn.train_polygon_head

  echo "=== Structured geometry + polygon IoU + area loss, seed ${seed} ==="
  EXPERIMENT_ID="polygon_struct_iou_area_s${seed}" SEED="$seed" \
    POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex_iou_area \
    POLYGON_IOU_WEIGHT=0.1 POLYGON_AREA_WEIGHT=0.1 POLYGON_FLIP=none \
    python -m src.polygon_rcnn.train_polygon_head
done

echo "Ablações poligonais concluídas. Resultados em: $RUNS_DIR"
