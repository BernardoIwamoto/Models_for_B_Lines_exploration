# Execução remota

Os scripts não treinam neste computador por causa do pedido do projeto; rode estes
comandos na máquina SSH com o repositório, dados e ambiente PyTorch/Detectron2 já
instalados. Execute da raiz do repositório.

## 1. Sincronizar e validar

Transfira este estado do repositório para o servidor (por Git/rsync conforme seu
fluxo), ative o ambiente Python de lá e confira a GPU:

```bash
cd /caminho/para/Models_for_B_Lines_exploration
git status --short
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem CUDA")'
python -m src.polygon_rcnn.validate_dataset --root data --output /tmp/blines_dataset_audit.json
```

A validação deve parar agora: existem duas linhas de treino com cinco vértices.
Confira-as nas fontes e corrija manualmente com base na annotation original. Não
remova um ponto por palpite. Depois rode novamente a validação. O treinamento e a
conversão YOLO só devem começar se ela terminar com código 0.

## 2. Converter os rótulos YOLO para detecção por caixas

Após a auditoria passar:

```bash
python -m src.polygon_yolo.detect.convert_to_detect
```

O conversor cria `data/yolo_detect/` e se recusa a substituir saídas diferentes.
Se já existir uma conversão antiga conflitante, preserve-a e escolha manualmente uma
pasta/dataset limpo antes de prosseguir.

## 3. Baselines Detectron2, três seeds

Cada execução recebe um diretório novo em `runs/`. Ajuste `RUNS_DIR` para um volume
com espaço para checkpoints. Rode cada linha em uma sessão separada ou em sequência:

```bash
export RUNS_DIR=/caminho/com/espaco/runs
for seed in 0 1 2; do
  EXPERIMENT_ID="mask_bbox_s${seed}" SEED="$seed" python -m src.polygon_rcnn.train_mask_rcnn
  EXPERIMENT_ID="faster_bbox_s${seed}" SEED="$seed" python -m src.polygon_rcnn.train_faster_rcnn
done
```

Para reproduzir primeiro a Polygon Head histórica, rode `legacy_relative` com loss
`vertex`:

```bash
for seed in 0 1 2; do
  EXPERIMENT_ID="polygon_legacy_s${seed}" SEED="$seed" POLYGON_REPRESENTATION=legacy_relative POLYGON_LOSS=vertex POLYGON_FLIP=none python -m src.polygon_rcnn.train_polygon_head
done
```

YOLOv11 bbox training uses a separate framework protocol; run it for the three-seed
baseline comparison:

```bash
for seed in 0 1 2; do
  EXPERIMENT_ID="yolo11n_bbox_s${seed}" SEED="$seed" python -m src.polygon_yolo.detect.train_yolo_detect
done
```

## 4. Próxima fase: parametrizações e loss

Depois de selecionar/reproduzir os baselines e executar as análises geométricas,
compare as representações relativas e estruturada com três seeds cada:

```bash
for seed in 0 1 2; do
  EXPERIMENT_ID="polygon_center_s${seed}" SEED="$seed" POLYGON_REPRESENTATION=center_relative POLYGON_LOSS=vertex POLYGON_FLIP=none python -m src.polygon_rcnn.train_polygon_head
  EXPERIMENT_ID="polygon_struct_s${seed}" SEED="$seed" POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex POLYGON_FLIP=none python -m src.polygon_rcnn.train_polygon_head
done
```

The structured representation predicts center, double-angle `(cos 2θ, sin 2θ)`,
length and two end widths; this makes orientation periodic modulo π. Then compare two
justified geometric loss additions, holding representation and seeds fixed:

```bash
for seed in 0 1 2; do
  EXPERIMENT_ID="polygon_struct_iou_s${seed}" SEED="$seed" POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex_iou POLYGON_IOU_WEIGHT=0.1 POLYGON_FLIP=none python -m src.polygon_rcnn.train_polygon_head
  EXPERIMENT_ID="polygon_struct_iou_area_s${seed}" SEED="$seed" POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex_iou_area POLYGON_IOU_WEIGHT=0.1 POLYGON_AREA_WEIGHT=0.1 POLYGON_FLIP=none python -m src.polygon_rcnn.train_polygon_head
done
```

The soft polygon IoU is a rasterized training approximation on the proposal ROI grid;
COCO polygon AP remains the selection metric. After selecting the best configuration,
compare `POLYGON_FLIP=none` with canonical reordering after horizontal flips:

```bash
for seed in 0 1 2; do
  EXPERIMENT_ID="polygon_struct_iou_area_flip_s${seed}" SEED="$seed" POLYGON_REPRESENTATION=structured POLYGON_LOSS=vertex_iou_area POLYGON_IOU_WEIGHT=0.1 POLYGON_AREA_WEIGHT=0.1 POLYGON_FLIP=canonical python -m src.polygon_rcnn.train_polygon_head
done
```

No origin auxiliary branch is trained: the available labels do not mark pleural origin.

## 5. Select the task metric checkpoint

For Detectron2, choose the metric that matches the output (Mask segmentation AP,
detector bbox AP, Polygon Head polygon segmentation AP):

```bash
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint runs/polygon_legacy_s0 segm_polygon/AP
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint runs/mask_bbox_s0 segm/AP
python -m src.polygon_rcnn.evaluation.common.select_best_checkpoint runs/faster_bbox_s0 bbox/AP
```

The new training scripts keep periodic checkpoints and save the configuration and
experiment manifest in each run folder. Checkpoint selection copies the best file;
it does not delete intermediate files. Repeat with each run ID. The YOLO best checkpoint
is `runs/yolo11n_bbox_s0/weights/best.pt`.

## 6. Avaliar cada modelo

Os diretórios de avaliação precisam ser novos. O exemplo de Faster R-CNN também cria
o JSON COCO de ground truth usado nas avaliações YOLO e bbox bias:

```bash
MODEL_PATH="$RUNS_DIR/mask_bbox_s0/model_best_segm_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/mask_bbox_s0/eval" python -m src.polygon_rcnn.evaluation.mask_rcnn.evaluate_coco
MODEL_PATH="$RUNS_DIR/faster_bbox_s0/model_best_bbox_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/faster_bbox_s0/eval" python -m src.polygon_rcnn.evaluation.faster_rcnn.evaluate_coco
GT_JSON="$RUNS_DIR/faster_bbox_s0/eval/blines_val_coco_format.json" MODEL_PATH="$RUNS_DIR/yolo11n_bbox_s0/weights/best.pt" EVAL_OUTPUT_DIR="$RUNS_DIR/yolo11n_bbox_s0/eval" python -m src.polygon_yolo.detect.evaluate_coco
```

## 7. Analisar geometria

Use um diretório de avaliação novo por execução. Exemplo para Polygon Head:

```bash
EXPERIMENT_ID=polygon_legacy_s0 POLYGON_REPRESENTATION=legacy_relative POLYGON_LOSS=vertex MODEL_PATH="$RUNS_DIR/polygon_legacy_s0/model_best_segm_polygon_AP.pth" EVAL_OUTPUT_DIR="$RUNS_DIR/polygon_legacy_s0/eval" python -m src.polygon_rcnn.evaluation.polygon_head.evaluate_coco
python -m src.polygon_rcnn.evaluation.geometry_metrics --gt "$RUNS_DIR/polygon_legacy_s0/eval/blines_val_coco_format.json" --predictions "$RUNS_DIR/polygon_legacy_s0/eval/coco_instances_results_polygon.json" --output "$RUNS_DIR/polygon_legacy_s0/geometry"
```

`geometry_metrics` grava CSV por instância, JSON de resumo e até três exemplos visuais
para cada categoria de erro. O matching é greedy por confiança com bbox IoU ≥ 0,50;
as métricas geométricas são condicionais a esse pareamento. Para o viés das caixas,
use o ground truth COCO e `coco_instances_results.json` do modelo:

```bash
python -m src.polygon_rcnn.evaluation.common.bbox_bias_analysis GT.json PREDICTIONS.json "$RUNS_DIR/bbox_bias/model" 0.5
```

O bootstrap compara predições fixas com reamostragens pareadas por imagem. Seus
intervalos não incluem a incerteza entre seeds de treino:

```bash
python -m src.polygon_rcnn.evaluation.bootstrap_coco --gt GT.json --prediction legacy=legacy_predictions.json --prediction structured=structured_predictions.json --task segm --replicates 1000 --seed 0 --output "$RUNS_DIR/bootstrap_segmentation.json"
```

Agregue resultados de runs já avaliados para obter média e desvio amostral entre
seeds. Repita `--result` com o mesmo nome de método:

```bash
python -m src.polygon_rcnn.evaluation.aggregate_runs --task segm_polygon \
  --result structured="$RUNS_DIR/polygon_struct_s0/eval/results.json" \
  --result structured="$RUNS_DIR/polygon_struct_s1/eval/results.json" \
  --result structured="$RUNS_DIR/polygon_struct_s2/eval/results.json" \
  --output "$RUNS_DIR/summary_structured.json"
```

Mantenha o manifesto, métricas, checkpoint selecionado, JSON de avaliação e resultados
geométricos de cada seed. Agregue média/desvio entre seeds separadamente dos intervalos
bootstrap por imagem: eles quantificam fontes diferentes de variação. `timing.json`
registra tempo de treino e tempo de inferência/avaliação Detectron2.
