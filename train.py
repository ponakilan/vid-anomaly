import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import FrameReconstructionModel
from dataset import ImageDataset, EmbeddingDataset, ImageEmbeddingDataset

root_dir = "data/UCSDped1/Train"
seq_len = 50

image_dataset = ImageDataset(
    root_dir=root_dir,
    seq_len=seq_len
)
embedding_dataset = EmbeddingDataset(
    embeddings_path="embeddings/embeddings_50.pkl"
)

dataset = ImageEmbeddingDataset(
    image_dataset=image_dataset,
    embedding_dataset=embedding_dataset
)
dataloader = DataLoader(dataset, batch_size=5)


model = FrameReconstructionModel()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in tqdm(enumerate(dataloader)):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

for epoch in range(10):
    print(f"Epoch {epoch}")
    avg_loss = train_one_epoch()