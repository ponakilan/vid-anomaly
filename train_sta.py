import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import StaModel
from dataset import ImageDataset, ImageEmbeddingDataset, EmbeddingGenerator

root_dir = sys.argv[2]
seq_len = 50

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

image_dataset = ImageDataset(
    root_dir=root_dir,
    seq_len=seq_len
)
embedding_dataset = EmbeddingGenerator(
    root_dir=root_dir,
    seq_len=seq_len
)
dataset = ImageEmbeddingDataset(
    image_dataset=image_dataset,
    embedding_dataset=embedding_dataset
)
dataloader = DataLoader(dataset, batch_size=12)

model = StaModel(device=device).to(device)

if len(sys.argv) > 3:
    model = torch.load(open(sys.argv[2], "rb"), weights_only=False).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in tqdm(enumerate(dataloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

for epoch in range(int(sys.argv[1])):
    print(f"Epoch {epoch}")
    avg_loss = train_one_epoch()

torch.save(model.cpu(), open("model_sta.pkl", "wb"))