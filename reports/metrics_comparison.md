# Validation metrics report

Values are mean ± sample standard deviation across seeds. COCO AP values and rate metrics are shown as percentages.
Object-level precision, recall, and F1 use each model's task at IoU 0.50. Image-presence metrics classify whether each image contains at least one B-line.
The confidence threshold is selected separately for each run to maximize F1 on validation; these are not independent test estimates.

## Baselines

### COCO bbox AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Mask R-CNN | 56.12 ± 1.74% | 91.23 ± 2.01% | 59.89 ± 4.13% |
| Faster R-CNN | 52.37 ± 2.38% | 87.95 ± 1.72% | 55.05 ± 4.23% |
| Polygon legacy | 49.70 ± 2.70% | 84.67 ± 1.67% | 50.60 ± 4.28% |
| YOLOv11n | 55.86 ± 0.99% | 85.26 ± 2.14% | 59.39 ± 2.66% |

### COCO keypoints AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Polygon legacy | 0.00 ± 0.00% | 0.02 ± 0.01% | 0.00 ± 0.00% |

### COCO segm AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Mask R-CNN | 42.88 ± 0.70% | 86.14 ± 0.98% | 35.98 ± 1.09% |

### COCO segm_polygon AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Polygon legacy | 14.36 ± 0.73% | 52.86 ± 0.70% | 1.63 ± 0.44% |

### Detecção por instância (segm, IoU 0,50; %) 

| Modelo | Precision | Recall | F1 | Limiar |
|---|---|---|---|---|
| Mask R-CNN | 79.01 ± 5.83% | 88.99 ± 4.79% | 83.47 ± 1.15% | 0.67 ± 0.10 |
| Polygon legacy | 62.09 ± 0.60% | 58.84 ± 4.46% | 60.36 ± 2.35% | 0.86 ± 0.03 |

### Detecção por instância (bbox, IoU 0,50; %) 

| Modelo | Precision | Recall | F1 | Limiar |
|---|---|---|---|---|
| Faster R-CNN | 82.96 ± 1.85% | 85.80 ± 1.33% | 84.33 ± 0.41% | 0.74 ± 0.08 |
| YOLOv11n | 79.20 ± 3.32% | 81.74 ± 3.98% | 80.45 ± 3.63% | 0.38 ± 0.04 |

### Presença de B-line por imagem (%)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Mask R-CNN | 97.78 ± 2.04% | 100.00 ± 0.00% | 97.78 ± 2.04% | 98.87 ± 1.04% |
| Faster R-CNN | 96.00 ± 1.33% | 100.00 ± 0.00% | 96.00 ± 1.33% | 97.96 ± 0.69% |
| Polygon legacy | 93.78 ± 2.78% | 100.00 ± 0.00% | 93.78 ± 2.78% | 96.77 ± 1.49% |
| YOLOv11n | 97.33 ± 1.33% | 100.00 ± 0.00% | 97.33 ± 1.33% | 98.65 ± 0.68% |

## Polygon ablations

### COCO bbox AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Center + vertex loss | 49.45 ± 2.79% | 84.23 ± 1.60% | 48.78 ± 3.85% |
| Structured + vertex loss | 50.10 ± 1.50% | 85.03 ± 0.17% | 53.16 ± 2.11% |
| Structured + IoU loss | 49.06 ± 2.20% | 85.39 ± 1.12% | 52.53 ± 5.37% |
| Structured + IoU + area loss | 50.66 ± 0.56% | 86.96 ± 0.38% | 53.41 ± 1.43% |

### COCO keypoints AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Center + vertex loss | 0.00 ± 0.00% | 0.02 ± 0.01% | 0.00 ± 0.00% |
| Structured + vertex loss | 0.14 ± 0.09% | 0.75 ± 0.49% | 0.00 ± 0.00% |
| Structured + IoU loss | 0.63 ± 0.12% | 2.32 ± 1.17% | 0.08 ± 0.11% |
| Structured + IoU + area loss | 0.73 ± 0.56% | 1.74 ± 1.10% | 0.48 ± 0.81% |

### COCO segm_polygon AP (%)

| Modelo | AP | AP50 | AP75 |
|---|---|---|---|
| Center + vertex loss | 14.63 ± 1.01% | 52.16 ± 3.65% | 1.66 ± 0.43% |
| Structured + vertex loss | 18.57 ± 1.54% | 57.98 ± 2.43% | 2.08 ± 1.09% |
| Structured + IoU loss | 17.82 ± 2.01% | 57.92 ± 0.58% | 5.87 ± 3.10% |
| Structured + IoU + area loss | 20.15 ± 2.44% | 60.08 ± 3.10% | 5.26 ± 2.11% |

### Detecção por instância (segm, IoU 0,50; %) 

| Modelo | Precision | Recall | F1 | Limiar |
|---|---|---|---|---|
| Center + vertex loss | 61.98 ± 3.26% | 57.68 ± 5.02% | 59.71 ± 3.89% | 0.87 ± 0.00 |
| Structured + vertex loss | 63.19 ± 3.37% | 64.93 ± 2.19% | 64.02 ± 2.30% | 0.84 ± 0.06 |
| Structured + IoU loss | 64.86 ± 4.65% | 63.19 ± 7.29% | 63.64 ± 1.96% | 0.87 ± 0.06 |
| Structured + IoU + area loss | 63.98 ± 2.82% | 67.83 ± 2.30% | 65.83 ± 2.32% | 0.86 ± 0.01 |

### Presença de B-line por imagem (%)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Center + vertex loss | 92.89 ± 2.04% | 100.00 ± 0.00% | 92.89 ± 2.04% | 96.31 ± 1.10% |
| Structured + vertex loss | 94.22 ± 2.78% | 100.00 ± 0.00% | 94.22 ± 2.78% | 97.01 ± 1.46% |
| Structured + IoU loss | 92.89 ± 6.58% | 100.00 ± 0.00% | 92.89 ± 6.58% | 96.23 ± 3.61% |
| Structured + IoU + area loss | 96.89 ± 0.77% | 100.00 ± 0.00% | 96.89 ± 0.77% | 98.42 ± 0.40% |

