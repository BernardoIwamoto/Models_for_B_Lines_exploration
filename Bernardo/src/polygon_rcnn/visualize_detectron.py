import cv2
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from register_dataset import register_blines


register_blines()

dataset = DatasetCatalog.get("blines_train")
metadata = MetadataCatalog.get("blines_train")

sample = dataset[0]

img = cv2.imread(sample["file_name"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

viz = Visualizer(
    img,
    metadata=metadata,
    scale=1.0,
)

out = viz.draw_dataset_dict(sample)

plt.figure(figsize=(8, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.show()