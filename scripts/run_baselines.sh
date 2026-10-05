#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNS_DIR:-}" ]]; then
  echo 'Defina RUNS_DIR para um diretório com espaço para checkpoints.' >&2
  exit 2
fi

if ! python -c 'import ultralytics' >/dev/null 2>&1; then
  echo 'Pacote ultralytics ausente no ambiente ativo.' >&2
  echo 'Ative a .venv e instale com: python -m pip install ultralytics' >&2
  exit 2
fi

python -m src.polygon_rcnn.validate_dataset --root data
python -m src.polygon_yolo.detect.convert_to_detect

for seed in 0 1 2; do
  echo "=== Mask R-CNN, seed ${seed} ==="
  experiment="mask_bbox_s${seed}"
  if [[ -f "$RUNS_DIR/$experiment/model_final.pth" ]]; then
    echo "=== Reutilizando checkpoint concluído: ${experiment} ==="
  elif [[ -e "$RUNS_DIR/$experiment" ]]; then
    echo "Run incompleta existe sem model_final.pth: $RUNS_DIR/$experiment" >&2
    exit 2
  else
    EXPERIMENT_ID="$experiment" SEED="$seed" python -m src.polygon_rcnn.train_mask_rcnn
  fi

  echo "=== Faster R-CNN, seed ${seed} ==="
  experiment="faster_bbox_s${seed}"
  if [[ -f "$RUNS_DIR/$experiment/model_final.pth" ]]; then
    echo "=== Reutilizando checkpoint concluído: ${experiment} ==="
  elif [[ -e "$RUNS_DIR/$experiment" ]]; then
    echo "Run incompleta existe sem model_final.pth: $RUNS_DIR/$experiment" >&2
    exit 2
  else
    EXPERIMENT_ID="$experiment" SEED="$seed" python -m src.polygon_rcnn.train_faster_rcnn
  fi

  echo "=== Polygon Head histórica, seed ${seed} ==="
  experiment="polygon_legacy_s${seed}"
  if [[ -f "$RUNS_DIR/$experiment/model_final.pth" ]]; then
    echo "=== Reutilizando checkpoint concluído: ${experiment} ==="
  elif [[ -e "$RUNS_DIR/$experiment" ]]; then
    echo "Run incompleta existe sem model_final.pth: $RUNS_DIR/$experiment" >&2
    exit 2
  else
    EXPERIMENT_ID="$experiment" SEED="$seed" \
      POLYGON_REPRESENTATION=legacy_relative POLYGON_LOSS=vertex POLYGON_FLIP=none \
      python -m src.polygon_rcnn.train_polygon_head
  fi

  echo "=== YOLOv11 bbox, seed ${seed} ==="
  experiment="yolo11n_bbox_s${seed}"
  if [[ -f "$RUNS_DIR/$experiment/weights/best.pt" ]]; then
    echo "=== Reutilizando checkpoint concluído: ${experiment} ==="
  elif [[ -e "$RUNS_DIR/$experiment" ]]; then
    echo "Run incompleta existe sem weights/best.pt: $RUNS_DIR/$experiment" >&2
    exit 2
  else
    EXPERIMENT_ID="$experiment" SEED="$seed" python -m src.polygon_yolo.detect.train_yolo_detect
  fi
done

bash scripts/evaluate_runs.sh baselines

echo "Baselines concluídos. Resultados em: $RUNS_DIR"
