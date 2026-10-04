# Execução na SSH

## 1. Copiar o pacote atualizado

No terminal local, transfira o pacote para o servidor:

```bash
scp /private/tmp/bline_research_changes.tar.gz USUARIO@HOST:/tmp/
```

Na SSH, rode:

```bash
cd ~/Models_for_B_Lines_exploration
tar -xzf /tmp/bline_research_changes.tar.gz -C .
source .venv/bin/activate
```

Troque `USUARIO` e `HOST` pelos dados da sua conexão SSH. Se o ambiente já estiver
ativo, pule o `source`. Use uma pasta nova por rodada para não colidir com execuções
anteriores; o exemplo de treino abaixo cria um nome com data e hora.

O pacote inclui as duas anotações normalizadas e `docs/annotation_corrections.json`,
que guarda as linhas originais. A simplificação altera a área de uma delas em 7,7%;
confira essa aproximação com a anotação original antes de usar resultados em artigo.

## 2. Treinar os baselines

Este script valida o dataset, converte as caixas YOLO e roda Mask R-CNN, Faster R-CNN,
Polygon Head histórica e YOLOv11 com três seeds cada. Se a validação falhar, o script
para antes de treinar.

```bash
export RUNS_DIR="$PWD/runs_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_baselines.sh
```

## 3. Rodar as novas ablações poligonais

Depois dos baselines, rode as representações centrada e estruturada, com ablações de
loss por vértice, IoU poligonal aproximada e área. Cada configuração usa três seeds.

```bash
bash scripts/run_polygon_ablation.sh
```

Para continuar a execução após desconectar da SSH, abra uma sessão `tmux` antes dos
comandos de treino (`tmux new -s blines`); reconecte com `tmux attach -t blines`.

## 4. Selecionar checkpoints e avaliar

Depois do treino, os comandos abaixo selecionam os checkpoints do seed 0. Repita
mudando `s0` para `s1` e `s2`. A seleção preserva os checkpoints intermediários.

```bash
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint "$RUNS_DIR/mask_bbox_s0" segm/AP
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint "$RUNS_DIR/faster_bbox_s0" bbox/AP
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint "$RUNS_DIR/polygon_legacy_s0" segm_polygon/AP
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint "$RUNS_DIR/polygon_struct_s0" segm_polygon/AP

MODEL_PATH="$RUNS_DIR/mask_bbox_s0/model_best_segm_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/mask_bbox_s0/eval" python -m src.polygon_rcnn.evaluation.mask_rcnn.evaluate_coco
MODEL_PATH="$RUNS_DIR/faster_bbox_s0/model_best_bbox_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/faster_bbox_s0/eval" python -m src.polygon_rcnn.evaluation.faster_rcnn.evaluate_coco
GT_JSON="$RUNS_DIR/faster_bbox_s0/eval/blines_val_coco_format.json" MODEL_PATH="$RUNS_DIR/yolo11n_bbox_s0/weights/best.pt" EVAL_OUTPUT_DIR="$RUNS_DIR/yolo11n_bbox_s0/eval" python -m src.polygon_yolo.detect.evaluate_coco
EXPERIMENT_ID=polygon_legacy_s0 POLYGON_REPRESENTATION=legacy_relative POLYGON_LOSS=vertex MODEL_PATH="$RUNS_DIR/polygon_legacy_s0/model_best_segm_polygon_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/polygon_legacy_s0/eval" python -m src.polygon_rcnn.evaluation.polygon_head.evaluate_coco
EXPERIMENT_ID=polygon_struct_s0 POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex MODEL_PATH="$RUNS_DIR/polygon_struct_s0/model_best_segm_polygon_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/polygon_struct_s0/eval" python -m src.polygon_rcnn.evaluation.polygon_head.evaluate_coco
```

Análise geométrica e bootstrap pareado por imagem:

```bash
python -m src.polygon_rcnn.evaluation.geometry_metrics --gt "$RUNS_DIR/polygon_legacy_s0/eval/blines_val_coco_format.json" --predictions "$RUNS_DIR/polygon_legacy_s0/eval/coco_instances_results_polygon.json" --output "$RUNS_DIR/polygon_legacy_s0/geometry"
python -m src.polygon_rcnn.evaluation.bootstrap_coco --gt "$RUNS_DIR/faster_bbox_s0/eval/blines_val_coco_format.json" --prediction legacy="$RUNS_DIR/polygon_legacy_s0/eval/coco_instances_results_polygon.json" --prediction structured="$RUNS_DIR/polygon_struct_s0/eval/coco_instances_results_polygon.json" --task segm --replicates 1000 --seed 0 --output "$RUNS_DIR/bootstrap_segmentation.json"
```

Bootstrap intervals are conditional on fixed predictions and do not include
between-seed training uncertainty. Keep those intervals separate from mean and sample
standard deviation across seeds.
