# Revisão bibliográfica para a próxima rodada

Esta revisão trata os links fornecidos como referências a verificar; artigos com outro
dataset ou outro alvo não são baselines diretamente comparáveis.

## LUS-BALD

O artigo descreve um conjunto de ultrassom pulmonar com 401 imagens de 152 pacientes,
instâncias de B-lines anotadas como pleural-based bounding boxes (PBBs), partições de
treino/validação anotadas e teste cego. O formato quadrilateral local é compatível com
a ideia de PBB, mas a contagem local (256/75/70) não demonstra proveniência byte a byte
da distribuição publicada. O artigo discute também as limitações de uma coorte pequena
e de anotações/referência humanas.

Fonte primária: [LUS-BALD, Scientific Data (2025)](https://www.nature.com/articles/s41597-025-05854-4).

## Supervisão da origem

Lucassen et al. investigam detecção e localização de B-lines. A localização de origem
pleural é um alvo distinto da segmentação da extensão inteira; no artigo, a supervisão
de origem é derivada de annotations apropriadas e representada como região circular
(raio de 4 mm) em parte dos experimentos. Isso motiva uma tarefa auxiliar apenas quando
existe uma definição verificável de origem nos rótulos. Os quadriláteros deste projeto
não rotulam essa origem, então não há label auxiliar confiável para treinar agora.

Fonte primária: [Lucassen et al., IEEE Journal of Biomedical and Health Informatics (2023), texto no PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10540221/).

## Loss com informação de fronteira

“Improved A-Line and B-Line Detection in Lung Ultrasound Using Deep Learning with
Boundary-Aware Dice Loss” propõe uma combinação de supervisão de segmentação e
fronteira. A hipótese transferível é ponderar explicitamente a qualidade da borda; a
arquitetura e o protocolo publicados não devem ser tratados como diretamente
comparáveis ao detector poligonal local. A implementação atual começa por uma loss de
IoU poligonal diferenciável aproximada e loss relativa de área, sem alegar equivalência
à Boundary-Aware Dice.

Fonte primária: [Abbasi et al., Bioengineering (2025)](https://www.mdpi.com/2306-5354/12/3/311).

## YOLO e PBB

O link IEEE fornecido (documento 11482811) não foi possível identificar como um artigo
de B-lines/PBBs. Não atribuímos a ele método ou resultado. Uma publicação primária
relacionada encontrada é Frontiers 2025, que descreve uma variante YOLOv5-PBB com
quatro vértices ordenados, loss de regressão Smooth-L1, métrica/IoU poligonal e NMS
específico; também discute YOLOv8-PBB e avaliação em dados próprios. É uma referência
para hipóteses de codificação/ordenação/supressão, não uma comparação numérica direta:
dataset e protocolo diferem.

Fonte primária: [Deep learning for accurate B-line detection and localization in lung ultrasound imaging, Frontiers in AI (2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1560523/full).

## Referência fornecida que não é relacionada

O preprint arXiv:2510.11390 é sobre interpretabilidade e mapas de conhecimento de LLMs
médicos; não apresenta metodologia substantiva para detecção/localização de B-lines.
Não deve entrar no related work técnico deste estudo.

Fonte primária: [arXiv:2510.11390](https://arxiv.org/abs/2510.11390).
