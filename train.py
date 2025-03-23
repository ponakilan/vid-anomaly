import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import FrameReconstructionModel
from dataset import ImageDataset, EmbeddingDataset, ImageEmbeddingDataset

root_dir = "data/UCSDped1/Train"
seq_len = 50

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

image_dataset = ImageDataset(
    root_dir=root_dir,
    seq_len=seq_len
)
embedding_dataset = EmbeddingDataset(
    embeddings_path="vid-anomaly/embeddings/embeddings_50.pkl"
)

dataset = ImageEmbeddingDataset(
    image_dataset=image_dataset,
    embedding_dataset=embedding_dataset
)
dataloader = DataLoader(dataset, batch_size=12)


model = FrameReconstructionModel(device=device).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in tqdm(enumerate(dataloader)):
        inputs, labels = data
        input, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

for epoch in range(10):
    print(f"Epoch {epoch}")
    avg_loss = train_one_epoch()

torch.save(model.cpu(), open("model.pkl", "wb"))