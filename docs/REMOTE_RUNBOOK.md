# Execução na Titan via SSH

## 1. Atualizar a correção do loader sem mesclar branches

Publique no GitHub, a partir do Mac, o commit que contém a correção do loader. Na Titan,
se `git pull --ff-only` falhar porque `main` e `origin/main` divergiram, não faça merge
nem apague os diretórios `runs_*` para iniciar o treino. Atualize a referência do GitHub
e copie para a árvore de trabalho somente os quatro arquivos corrigidos e `.gitignore`.
Isso mantém as outras alterações locais no lugar:

```bash
cd ~/Models_for_B_Lines_exploration
git fetch origin
git status --short -- .gitignore \
  src/polygon_rcnn/evaluation/common/hooks.py \
  src/polygon_rcnn/train_mask_rcnn.py \
  src/polygon_rcnn/train_faster_rcnn.py \
  src/polygon_rcnn/train_polygon_head.py
git restore --source=origin/main --worktree -- \
  .gitignore \
  src/polygon_rcnn/evaluation/common/hooks.py \
  src/polygon_rcnn/train_mask_rcnn.py \
  src/polygon_rcnn/train_faster_rcnn.py \
  src/polygon_rcnn/train_polygon_head.py
grep -n "build_loss_eval_mapper" \
  src/polygon_rcnn/evaluation/common/hooks.py \
  src/polygon_rcnn/train_mask_rcnn.py
```

O `grep` deve mostrar a função e seu uso no Mask R-CNN. Antes de executar `git restore`,
confira o resultado de `git status`: se qualquer um dos arquivos listados estiver
modificado, preserve essa alteração antes de continuar.

## 2. Treinar os baselines

Este script valida o dataset, converte as caixas YOLO e roda Mask R-CNN, Faster R-CNN,
Polygon Head histórica e YOLOv11 com três seeds cada. Se a validação falhar, o script
para antes de treinar.

Abra uma sessão persistente:

```bash
tmux new -As blines
```

Dentro da sessão `tmux`, rode:

```bash
cd ~/Models_for_B_Lines_exploration
source .venv/bin/activate
export RUNS_DIR="$PWD/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNS_DIR"
set -o pipefail
bash scripts/run_baselines.sh 2>&1 | tee "$RUNS_DIR/baselines.log"
```

Para deixar treinando em segundo plano, pressione `Ctrl-b` e depois `d`. Para voltar
ao processo, rode `tmux attach -t blines`. O `RUNS_DIR` novo evita colisões com
experimentos anteriores; mantenha essa mesma variável na sessão para as etapas
seguintes.

## 3. Rodar as novas ablações poligonais

Depois dos baselines, rode as representações centrada e estruturada, com ablações de
loss por vértice, IoU poligonal aproximada e área. Cada configuração usa três seeds.

```bash
bash scripts/run_polygon_ablation.sh 2>&1 | tee "$RUNS_DIR/polygon_ablation.log"
```

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
