#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNS_DIR:-}" || -z "${REFERENCE_RUNS_DIR:-}" ]]; then
  echo 'Defina RUNS_DIR (saída nova) e REFERENCE_RUNS_DIR (controle estruturado sem pré-processamento, seeds 0–2).' >&2
  exit 2
fi

RUNS_DIR="$(mkdir -p "$RUNS_DIR" && cd "$RUNS_DIR" && pwd)"
REFERENCE_RUNS_DIR="$(cd "$REFERENCE_RUNS_DIR" && pwd)"
PREVIOUS_RUNS_DIR="${PREVIOUS_RUNS_DIR:-}"
if [[ -n "$PREVIOUS_RUNS_DIR" ]]; then
  PREVIOUS_RUNS_DIR="$(cd "$PREVIOUS_RUNS_DIR" && pwd)"
  if [[ "$PREVIOUS_RUNS_DIR" == "$RUNS_DIR" ]]; then
    echo 'PREVIOUS_RUNS_DIR e RUNS_DIR precisam ser pastas diferentes.' >&2
    exit 2
  fi
fi
PREPROCESSING_DATA_ROOT="${PREPROCESSING_DATA_ROOT:-$PWD/data_preprocessed}"
PREPROCESSING_DATA_ROOT="$(mkdir -p "$PREPROCESSING_DATA_ROOT" && cd "$PREPROCESSING_DATA_ROOT" && pwd)"
variants=(clahe nlm unsharp canny nlm_clahe)

for seed in 0 1 2; do
  if [[ ! -f "$REFERENCE_RUNS_DIR/polygon_struct_iou_area_s${seed}/eval/metrics_summary.json" ]]; then
    echo "Faltam métricas do controle para seed ${seed}: $REFERENCE_RUNS_DIR/polygon_struct_iou_area_s${seed}/eval/metrics_summary.json" >&2
    exit 2
  fi
done

for variant in "${variants[@]}"; do
  for seed in 0 1 2; do
    run_id="polygon_struct_iou_area_${variant}_s${seed}"
    if [[ -e "$RUNS_DIR/$run_id" && ! -f "$RUNS_DIR/$run_id/eval/metrics_summary.json" ]]; then
      echo "Run de saída incompleta já existe: $RUNS_DIR/$run_id. Escolha um RUNS_DIR novo." >&2
      exit 2
    fi
  done
done

if [[ -n "$PREVIOUS_RUNS_DIR" ]]; then
  for variant in "${variants[@]}"; do
    for seed in 0 1 2; do
      run_id="polygon_struct_iou_area_${variant}_s${seed}"
      old_run="$PREVIOUS_RUNS_DIR/$run_id"
      if [[ -d "$old_run" && ! -f "$old_run/eval/metrics_summary.json" && ! -f "$old_run/model_final.pth" ]]; then
        echo "=== Removendo apenas snapshots do treino interrompido: ${run_id} ==="
        find "$old_run" -maxdepth 1 -type f -name 'model_[0-9]*.pth' -delete
      fi
    done
  done
fi

prune_numbered_snapshots() {
  local run_dir="$1"
  local metrics_file="${2:-$run_dir/eval/metrics_summary.json}"
  if [[ -f "$metrics_file" && -f "$run_dir/model_best_segm_polygon_AP.pth" ]]; then
    find "$run_dir" -maxdepth 1 -type f -name 'model_[0-9]*.pth' -delete
    rm -f "$run_dir/model_final.pth"
  fi
}

evaluate_polygon_run() {
  local run_id="$1"
  local dataset_root="$2"
  local checkpoint_dir="$3"
  local output_run_dir="$4"
  local best_checkpoint="$checkpoint_dir/model_best_segm_polygon_AP.pth"
  if [[ ! -f "$best_checkpoint" ]]; then
    python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint \
      "$checkpoint_dir" segm_polygon/AP
  fi
  BLINES_DATA_ROOT="$dataset_root" \
  EXPERIMENT_ID="$run_id" \
  POLYGON_REPRESENTATION=structured \
  POLYGON_LOSS=vertex_iou_area \
  MODEL_PATH="$best_checkpoint" \
  EVAL_OUTPUT_DIR="$output_run_dir/eval" \
    python -m src.polygon_rcnn.evaluation.polygon_head.evaluate_coco
}

for variant in "${variants[@]}"; do
  dataset_root="$PREPROCESSING_DATA_ROOT/$variant"
  if [[ -d "$dataset_root" ]]; then
    if ! grep -q "\"variant\": \"$variant\"" "$dataset_root/preprocessing.json" 2>/dev/null; then
      echo "Dataset processado existente não corresponde a ${variant}: $dataset_root" >&2
      exit 2
    fi
    echo "=== Reutilizando dataset processado: ${variant} ==="
  else
    echo "=== Preparando dados: ${variant} ==="
    python -m src.polygon_rcnn.prepare_preprocessing_dataset \
      --source data --output "$dataset_root" --variant "$variant"
  fi
  python -m src.polygon_rcnn.validate_dataset --root "$dataset_root"

  if [[ -n "$PREVIOUS_RUNS_DIR" ]]; then
    for seed in 0 1 2; do
      run_id="polygon_struct_iou_area_${variant}_s${seed}"
      old_run="$PREVIOUS_RUNS_DIR/$run_id"
      new_run="$RUNS_DIR/$run_id"
      [[ -d "$old_run" ]] || continue
      if [[ -f "$old_run/eval/metrics_summary.json" ]]; then
        mkdir -p "$new_run/eval"
        cp "$old_run/eval/metrics_summary.json" "$new_run/eval/metrics_summary.json"
        if [[ -f "$old_run/experiment.json" ]]; then
          cp "$old_run/experiment.json" "$new_run/experiment.json"
        fi
        prune_numbered_snapshots "$old_run"
        echo "=== Métricas concluídas reutilizadas: ${run_id} ==="
      elif [[ -f "$old_run/model_final.pth" ]]; then
        echo "=== Avaliando treino completo já existente: ${run_id} ==="
        mkdir -p "$new_run"
        if [[ -f "$old_run/experiment.json" ]]; then
          cp "$old_run/experiment.json" "$new_run/experiment.json"
        fi
        evaluate_polygon_run "$run_id" "$dataset_root" "$old_run" "$new_run"
        prune_numbered_snapshots "$old_run" "$new_run/eval/metrics_summary.json"
      else
        echo "=== Treino interrompido será refeito em RUNS_DIR novo: ${run_id} ==="
      fi
    done
  fi

  for seed in 0 1 2; do
    run_id="polygon_struct_iou_area_${variant}_s${seed}"
    new_run="$RUNS_DIR/$run_id"
    if [[ -f "$new_run/eval/metrics_summary.json" ]]; then
      echo "=== Run já avaliada; pulando treino: ${run_id} ==="
      continue
    fi
    if [[ -e "$new_run" ]]; then
      echo "Run de saída já existe sem métricas: $new_run. Escolha um RUNS_DIR novo." >&2
      exit 2
    fi
    echo "=== Treinando structured + IoU + area + ${variant}, seed ${seed} ==="
    RUNS_DIR="$RUNS_DIR" \
    BLINES_DATA_ROOT="$dataset_root" \
    IMAGE_PREPROCESSING="$variant" \
    EXPERIMENT_ID="$run_id" \
    SEED="$seed" \
    POLYGON_REPRESENTATION=structured \
    POLYGON_LOSS=vertex_iou_area \
    POLYGON_IOU_WEIGHT=0.1 \
    POLYGON_AREA_WEIGHT=0.1 \
    POLYGON_FLIP=none \
      python -m src.polygon_rcnn.train_polygon_head

    echo "=== Avaliando e liberando snapshots intermediários: ${run_id} ==="
    evaluate_polygon_run "$run_id" "$dataset_root" "$new_run" "$new_run"
    prune_numbered_snapshots "$new_run"
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
