from dataset import ImageDataset

dataset = ImageDataset(root_dir="data/UCSDped1/Train")
print(dataset[0].shape)