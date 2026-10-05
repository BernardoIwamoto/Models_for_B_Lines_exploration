#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNS_DIR:-}" ]]; then
  echo 'Defina RUNS_DIR para a pasta que contém os experimentos.' >&2
  exit 2
fi

MODE="${1:-}"
if [[ "$MODE" != "baselines" && "$MODE" != "polygon_ablations" ]]; then
  echo 'Uso: bash scripts/evaluate_runs.sh baselines|polygon_ablations' >&2
  exit 2
fi

evaluate_mask() {
  local experiment="$1"
  if [[ -f "$RUNS_DIR/$experiment/eval/metrics_summary.json" ]]; then
    echo "--- Já avaliado; reutilizando métricas: ${experiment} ---"
    return
  fi
  echo "--- Avaliando Mask R-CNN: ${experiment} ---"
  python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint \
    "$RUNS_DIR/$experiment" segm/AP
  MODEL_PATH="$RUNS_DIR/$experiment/model_best_segm_AP.pth" \
    EVAL_OUTPUT_DIR="$RUNS_DIR/$experiment/eval" \
    python -m src.polygon_rcnn.evaluation.mask_rcnn.evaluate_coco
}

evaluate_faster() {
  local experiment="$1"
  if [[ -f "$RUNS_DIR/$experiment/eval/metrics_summary.json" ]]; then
    echo "--- Já avaliado; reutilizando métricas: ${experiment} ---"
    return
  fi
  echo "--- Avaliando Faster R-CNN: ${experiment} ---"
  python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint \
    "$RUNS_DIR/$experiment" bbox/AP
  MODEL_PATH="$RUNS_DIR/$experiment/model_best_bbox_AP.pth" \
    EVAL_OUTPUT_DIR="$RUNS_DIR/$experiment/eval" \
    python -m src.polygon_rcnn.evaluation.faster_rcnn.evaluate_coco
}

evaluate_yolo() {
  local experiment="$1"
  local gt_json="$2"
  if [[ -f "$RUNS_DIR/$experiment/eval/metrics_summary.json" ]]; then
    echo "--- Já avaliado; reutilizando métricas: ${experiment} ---"
    return
  fi
  echo "--- Avaliando YOLO: ${experiment} ---"
  GT_JSON="$gt_json" \
    MODEL_PATH="$RUNS_DIR/$experiment/weights/best.pt" \
    EVAL_OUTPUT_DIR="$RUNS_DIR/$experiment/eval" \
    python -m src.polygon_yolo.detect.evaluate_coco
}

evaluate_polygon() {
  local experiment="$1"
  local representation="$2"
  local loss="$3"
  if [[ -f "$RUNS_DIR/$experiment/eval/metrics_summary.json" ]]; then
    echo "--- Já avaliado; reutilizando métricas: ${experiment} ---"
    return
  fi
  echo "--- Avaliando Polygon Head: ${experiment} ---"
  python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint \
    "$RUNS_DIR/$experiment" segm_polygon/AP
  EXPERIMENT_ID="$experiment" \
    POLYGON_REPRESENTATION="$representation" \
    POLYGON_LOSS="$loss" \
    MODEL_PATH="$RUNS_DIR/$experiment/model_best_segm_polygon_AP.pth" \
    EVAL_OUTPUT_DIR="$RUNS_DIR/$experiment/eval" \
    python -m src.polygon_rcnn.evaluation.polygon_head.evaluate_coco
}

if [[ "$MODE" == "baselines" ]]; then
  for seed in 0 1 2; do
    evaluate_mask "mask_bbox_s${seed}"
    evaluate_faster "faster_bbox_s${seed}"
    evaluate_polygon "polygon_legacy_s${seed}" legacy_relative vertex
    evaluate_yolo "yolo11n_bbox_s${seed}" \
      "$RUNS_DIR/faster_bbox_s${seed}/eval/blines_val_coco_format.json"
  done
else
  for seed in 0 1 2; do
    evaluate_polygon "polygon_center_s${seed}" center_relative vertex
    evaluate_polygon "polygon_struct_s${seed}" structured vertex
    evaluate_polygon "polygon_struct_iou_s${seed}" structured vertex_iou
    evaluate_polygon "polygon_struct_iou_area_s${seed}" structured vertex_iou_area
  done
fi

python -m src.polygon_rcnn.evaluation.aggregate_validation_metrics \
  --runs-dir "$RUNS_DIR" --suite "$MODE" --seeds 0 1 2

echo "Avaliações e métricas concluídas. Consulte eval/metrics_summary.json em cada run sob: $RUNS_DIR"
