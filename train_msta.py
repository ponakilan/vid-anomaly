import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.models import FrameReconstructionModel
from core.dataset import ImageDataset, ImageEmbeddingDataset, EmbeddingDataset


def main(root_dir, embedding_file, seq_len, epochs, save_file):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image_dataset = ImageDataset(
        root_dir=root_dir,
        seq_len=seq_len
    )
    embedding_dataset = EmbeddingDataset(
        embeddings_path=embedding_file
    )
    dataset = ImageEmbeddingDataset(
        image_dataset=image_dataset,
        embedding_dataset=embedding_dataset
    )
    dataloader = DataLoader(dataset, batch_size=12)

    model = FrameReconstructionModel(device=device).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


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

    for epoch in range(int(epochs)):
        print(f"Epoch {epoch}")
        avg_loss = train_one_epoch()

    torch.save(model.cpu(), open(save_file, "wb"))

if __name__ == "__main__":
    root_dir = sys.argv[1]
    embedding_file = sys.argv[2]
    seq_len = int(sys.argv[3])
    epochs = int(sys.argv[4])
    save_file = sys.argv[5]

    main(root_dir, embedding_file, seq_len, epochs, save_file)
