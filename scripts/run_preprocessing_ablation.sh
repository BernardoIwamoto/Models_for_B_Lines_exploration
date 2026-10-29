#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNS_DIR:-}" || -z "${REFERENCE_RUNS_DIR:-}" ]]; then
  echo 'Defina RUNS_DIR (nova saída) e REFERENCE_RUNS_DIR (controle estruturado sem pré-processamento, seeds 0–2).' >&2
  exit 2
fi

RUNS_DIR="$(mkdir -p "$RUNS_DIR" && cd "$RUNS_DIR" && pwd)"
REFERENCE_RUNS_DIR="$(cd "$REFERENCE_RUNS_DIR" && pwd)"
PREPROCESSING_DATA_ROOT="${PREPROCESSING_DATA_ROOT:-$PWD/data_preprocessed}"
PREPROCESSING_DATA_ROOT="$(mkdir -p "$PREPROCESSING_DATA_ROOT" && cd "$PREPROCESSING_DATA_ROOT" && pwd)"
variants=(clahe nlm unsharp canny nlm_clahe)

for seed in 0 1 2; do
  if [[ ! -f "$REFERENCE_RUNS_DIR/polygon_struct_iou_area_s${seed}/eval/metrics_summary.json" ]]; then
    echo "Faltam métricas de referência para seed ${seed}: $REFERENCE_RUNS_DIR/polygon_struct_iou_area_s${seed}/eval/metrics_summary.json" >&2
    exit 2
  fi
done

for variant in "${variants[@]}"; do
  for seed in 0 1 2; do
    run_id="polygon_struct_iou_area_${variant}_s${seed}"
    if [[ -e "$RUNS_DIR/$run_id" ]]; then
      echo "A saída já existe: $RUNS_DIR/$run_id. Escolha outro RUNS_DIR." >&2
      exit 2
    fi
  done
  if [[ -e "$PREPROCESSING_DATA_ROOT/$variant" ]]; then
    echo "O dataset processado já existe: $PREPROCESSING_DATA_ROOT/$variant. Escolha outro PREPROCESSING_DATA_ROOT." >&2
    exit 2
  fi
done

for variant in "${variants[@]}"; do
  dataset_root="$PREPROCESSING_DATA_ROOT/$variant"
  echo "=== Preparando dados: ${variant} ==="
  python -m src.polygon_rcnn.prepare_preprocessing_dataset \
    --source data \
    --output "$dataset_root" \
    --variant "$variant"
  python -m src.polygon_rcnn.validate_dataset --root "$dataset_root"

  for seed in 0 1 2; do
    echo "=== Structured + IoU + area + ${variant}, seed ${seed} ==="
    RUNS_DIR="$RUNS_DIR" \
    BLINES_DATA_ROOT="$dataset_root" \
    IMAGE_PREPROCESSING="$variant" \
    EXPERIMENT_ID="polygon_struct_iou_area_${variant}_s${seed}" \
    SEED="$seed" \
    POLYGON_REPRESENTATION=structured \
    POLYGON_LOSS=vertex_iou_area \
    POLYGON_IOU_WEIGHT=0.1 \
    POLYGON_AREA_WEIGHT=0.1 \
    POLYGON_FLIP=none \
    python -m src.polygon_rcnn.train_polygon_head
  done
done

RUNS_DIR="$RUNS_DIR" PREPROCESSING_DATA_ROOT="$PREPROCESSING_DATA_ROOT" \
  bash scripts/evaluate_runs.sh preprocessing_ablation

python -m src.polygon_rcnn.evaluation.compare_preprocessing \
  --reference-runs-dir "$REFERENCE_RUNS_DIR" \
  --runs-dir "$RUNS_DIR" \
  --output-dir "$RUNS_DIR/reports/preprocessing_ablation" \
  --seeds 0 1 2

echo "Experimentos de pré-processamento concluídos. Resultados: $RUNS_DIR"
echo "Tabela: $RUNS_DIR/reports/preprocessing_ablation/preprocessing_comparison.md"
echo "Gráficos: $RUNS_DIR/reports/preprocessing_ablation/*.png"
