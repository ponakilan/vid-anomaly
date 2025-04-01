from core.dataset import EmbeddingGenerator

dataset = EmbeddingGenerator(
    "data/IPAD_dataset/R01/training/frames",
    50
)

print(dataset[0])