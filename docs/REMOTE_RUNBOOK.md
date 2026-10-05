# Execução na Titan via SSH

## 1. Atualizar a correção do loader sem mesclar branches

Publique no GitHub, a partir do Mac, o commit com as correções do loader e das métricas.
Na Titan, se `git pull --ff-only` falhar porque `main` e `origin/main` divergiram, não
faça merge nem apague os diretórios `runs_*`. O status que você enviou mostra dois
scripts locais não rastreados; faça cópia deles antes de atualizá-los. Depois busque os
arquivos novos/corrigidos do GitHub, sem substituir suas anotações, relatório ou saídas:

```bash
cd ~/Models_for_B_Lines_exploration
git fetch origin
mkdir -p /tmp/blines_script_backup
cp scripts/run_baselines.sh /tmp/blines_script_backup/
cp scripts/run_polygon_ablation.sh /tmp/blines_script_backup/
git restore --source=origin/main --worktree -- \
  .gitignore \
  scripts/run_baselines.sh \
  scripts/run_polygon_ablation.sh \
  scripts/evaluate_runs.sh \
  src/polygon_rcnn/evaluation/common/hooks.py \
  src/polygon_rcnn/evaluation/common/metrics_report.py \
  src/polygon_rcnn/evaluation/aggregate_validation_metrics.py \
  src/polygon_rcnn/train_mask_rcnn.py \
  src/polygon_rcnn/train_faster_rcnn.py \
  src/polygon_rcnn/train_polygon_head.py \
  src/polygon_rcnn/evaluation/mask_rcnn/evaluate_coco.py \
  src/polygon_rcnn/evaluation/faster_rcnn/evaluate_coco.py \
  src/polygon_rcnn/evaluation/polygon_head/evaluate_coco.py \
  src/polygon_yolo/detect/evaluate_coco.py \
  src/polygon_rcnn/evaluation/common/coco_threshold_plots.py
grep -n "build_loss_eval_mapper" \
  src/polygon_rcnn/evaluation/common/hooks.py \
  src/polygon_rcnn/train_mask_rcnn.py
```

O `grep` deve mostrar a função e seu uso no Mask R-CNN. Os backups dos scripts locais
ficam em `/tmp/blines_script_backup`.

## 2. Treinar os baselines

Este script valida o dataset, converte as caixas YOLO e roda Mask R-CNN, Faster R-CNN,
Polygon Head histórica e YOLOv11 com três seeds cada. Se a validação falhar, o script
para antes de treinar. Antes da sessão `tmux`, instale e confira o pacote YOLO no mesmo
ambiente virtual usado para os modelos Detectron2:

```bash
source .venv/bin/activate
python -m pip install ultralytics
python -c 'from ultralytics import YOLO; print("ultralytics OK")'
```

O script também faz essa verificação antes do primeiro treino e para com uma instrução
de instalação se `ultralytics` estiver ausente.

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
ao processo, rode `tmux attach -t blines`. Para continuar uma suíte que parou, exporte
o mesmo `RUNS_DIR` e rode o script novamente: checkpoints concluídos serão reutilizados
e os experimentos ausentes serão treinados. Para uma repetição independente, use um
`RUNS_DIR` novo. Mantenha a variável na sessão para as etapas seguintes.

## 3. Rodar as novas ablações poligonais

Depois dos baselines, rode as representações centrada e estruturada, com ablações de
loss por vértice, IoU poligonal aproximada e área. Cada configuração usa três seeds.

```bash
bash scripts/run_polygon_ablation.sh 2>&1 | tee "$RUNS_DIR/polygon_ablation.log"
```

## 4. Arquivos de métricas

Ao terminar cada suíte, o script seleciona o melhor checkpoint de cada run e executa
a avaliação COCO em todos os seeds e variantes. Cada run recebe `eval/results.json`,
`eval/metrics_summary.json` e gráficos das curvas. Os agregados entre os três seeds
ficam em `$RUNS_DIR/aggregated_baselines_metrics.json` e
`$RUNS_DIR/aggregated_polygon_ablations_metrics.json`, sem uma suíte sobrescrever a outra.

O resumo inclui AP/AP50/AP75, precisão, recall e F1 por detecção. Precisão, recall e F1
usam matching COCO na tarefa correspondente (bbox para Faster/YOLO e segmentação para
Mask/Polygon), IoU 0,50 e o limiar de confiança que maximiza F1 na validação. A acurácia
é definida separadamente como classificação binária por imagem (há/não há B-line), com
TP/TN/FP/FN. As métricas são da validação; o limiar escolhido na validação não constitui
uma estimativa independente de teste.

`run_baselines.sh` gera esses arquivos para Mask R-CNN, Faster R-CNN, YOLO e Polygon
Head histórica em todos os seeds. `run_polygon_ablation.sh` também avalia todas as
representações e losses testadas, em todos os seeds.

Os resultados por detecção e imagem são comparáveis sob a definição acima. Os valores
COCO de bbox e segmentação medem tarefas distintas e devem ser comparados dentro da
mesma tarefa.

## 5. Análise geométrica opcional

Depois das avaliações, estes comandos analisam a geometria e comparam as variantes de
polígono no seed 0:

Análise geométrica e bootstrap pareado por imagem:

```bash
python -m src.polygon_rcnn.evaluation.geometry_metrics --gt "$RUNS_DIR/polygon_legacy_s0/eval/blines_val_coco_format.json" --predictions "$RUNS_DIR/polygon_legacy_s0/eval/coco_instances_results_polygon.json" --output "$RUNS_DIR/polygon_legacy_s0/geometry"
python -m src.polygon_rcnn.evaluation.bootstrap_coco --gt "$RUNS_DIR/faster_bbox_s0/eval/blines_val_coco_format.json" --prediction legacy="$RUNS_DIR/polygon_legacy_s0/eval/coco_instances_results_polygon.json" --prediction structured="$RUNS_DIR/polygon_struct_s0/eval/coco_instances_results_polygon.json" --task segm --replicates 1000 --seed 0 --output "$RUNS_DIR/bootstrap_segmentation.json"
```

Bootstrap intervals are conditional on fixed predictions and do not include
between-seed training uncertainty. Keep those intervals separate from mean and sample
standard deviation across seeds.
