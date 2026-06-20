from detectron2.data import DatasetCatalog

from register_dataset import register_blines


register_blines()

dataset = DatasetCatalog.get("blines_train")

print(f"Images: {len(dataset)}")

sample = dataset[0]

print("\nKeys:")
print(sample.keys())

print("\nImage:")
print(sample["file_name"])

print("\nAnnotations:")
print(sample["annotations"][0])