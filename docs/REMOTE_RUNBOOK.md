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

Depois que os baselines e as ablações tiverem seus arquivos `eval/metrics_summary.json`,
gere tabelas legíveis e gráficos comparativos sem repetir treinos nem inferências:

```bash
python -m src.polygon_rcnn.evaluation.build_metrics_report --runs-dir "$RUNS_DIR"
```

Os arquivos ficam em `$RUNS_DIR/reports/`: `metrics_comparison.md`, dois CSVs e gráficos
PNG de presença por imagem, métricas por detecção e AP COCO por tarefa. Como `runs_*` é
ignorado pelo Git, copie essa pasta da Titan via `scp` para abrir os gráficos no Mac.

## 5. Análises adicionais sem novo treinamento

Rode depois que a avaliação tiver gerado os arquivos `coco_instances_results*.json`.
Os exemplos abaixo usam seed 0 para reduzir o custo; para relatar a análise geométrica
como resultado final, repita nos seeds 1 e 2 e agregue por seed.

Métricas geométricas das cinco variantes poligonais (inclui matching por bbox IoU
`>= 0.50`, erro de vértices, IoU/área/orientação e polígonos inválidos):

```bash
for model in polygon_legacy polygon_center polygon_struct polygon_struct_iou polygon_struct_iou_area; do
  python -m src.polygon_rcnn.evaluation.geometry_metrics \
    --gt "$RUNS_DIR/${model}_s0/eval/blines_val_coco_format.json" \
    --predictions "$RUNS_DIR/${model}_s0/eval/coco_instances_results_polygon.json" \
    --output "$RUNS_DIR/${model}_s0/geometry"
done
```

Viés das caixas dos três detectores baselines, com o mesmo conjunto de validação:

```bash
python -m src.polygon_rcnn.evaluation.common.bbox_bias_analysis "$RUNS_DIR/mask_bbox_s0/eval/blines_val_coco_format.json" "$RUNS_DIR/mask_bbox_s0/eval/coco_instances_results.json" "$RUNS_DIR/mask_bbox_s0/bbox_bias"
python -m src.polygon_rcnn.evaluation.common.bbox_bias_analysis "$RUNS_DIR/faster_bbox_s0/eval/blines_val_coco_format.json" "$RUNS_DIR/faster_bbox_s0/eval/coco_instances_results.json" "$RUNS_DIR/faster_bbox_s0/bbox_bias"
python -m src.polygon_rcnn.evaluation.common.bbox_bias_analysis "$RUNS_DIR/yolo11n_bbox_s0/eval/blines_val_coco_format.json" "$RUNS_DIR/yolo11n_bbox_s0/eval/coco_instances_results.json" "$RUNS_DIR/yolo11n_bbox_s0/bbox_bias"
```

Bootstrap pareado por imagem para AP, AP50 e AP75 das variantes poligonais, usando as
mesmas reamostragens para cada método:

```bash
python -m src.polygon_rcnn.evaluation.bootstrap_coco \
  --gt "$RUNS_DIR/polygon_legacy_s0/eval/blines_val_coco_format.json" \
  --prediction legacy="$RUNS_DIR/polygon_legacy_s0/eval/coco_instances_results_polygon.json" \
  --prediction center="$RUNS_DIR/polygon_center_s0/eval/coco_instances_results_polygon.json" \
  --prediction structured="$RUNS_DIR/polygon_struct_s0/eval/coco_instances_results_polygon.json" \
  --prediction structured_iou="$RUNS_DIR/polygon_struct_iou_s0/eval/coco_instances_results_polygon.json" \
  --prediction structured_iou_area="$RUNS_DIR/polygon_struct_iou_area_s0/eval/coco_instances_results_polygon.json" \
  --task segm --replicates 1000 --seed 0 \
  --output "$RUNS_DIR/bootstrap_polygon_seed0.json"
```

Os intervalos bootstrap são condicionais às predições/checkpoints fixos; não incluem
incerteza entre seeds. Para a comparação com baselines de máscara ou caixa, use os
arquivos de predição da tarefa correspondente e a mesma amostra pareada. Mantenha
esses intervalos separados da média e do desvio-padrão entre seeds.
