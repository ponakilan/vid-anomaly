import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.models import FrameReconstructionModel
from core.dataset import ImageDataset, ImageEmbeddingDataset, EmbeddingDataset


def main(root_dir_train, root_dir_test, embedding_file_train, embedding_file_test, seq_len_train, seq_len_test, epochs, save_file):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image_dataset_train = ImageDataset(
        root_dir=root_dir_train,
        seq_len=seq_len_train
    )
    embedding_dataset_train = EmbeddingDataset(
        embeddings_path=embedding_file_train
    )
    dataset_train = ImageEmbeddingDataset(
        image_dataset=image_dataset_train,
        embedding_dataset=embedding_dataset_train
    )
    dataloader_train = DataLoader(dataset_train, batch_size=4)

    image_dataset_test = ImageDataset(
        root_dir=root_dir_test,
        seq_len=seq_len_test
    )
    embedding_dataset_test = EmbeddingDataset(
        embeddings_path=embedding_file_test
    )
    dataset_test = ImageEmbeddingDataset(
        image_dataset=image_dataset_test,
        embedding_dataset=embedding_dataset_test
    )
    dataloader_test = DataLoader(dataset_test, batch_size=2)

    model = FrameReconstructionModel(device=device).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    def train_one_epoch():
        running_loss = 0.
        skip_sequences = [2, 3, 13, 17, 18, 20, 21, 22, 23, 31]

        for i, data in tqdm(enumerate(dataloader_train)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        for i, data in tqdm(enumerate(dataloader_test)):

            if i in skip_sequences:
                continue

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        return running_loss / (len(dataloader_train) + len(dataloader_test))

    for epoch in range(int(epochs)):
        print(f"Epoch {epoch}")
        avg_loss = train_one_epoch()
        print(f"Epoch loss: {avg_loss}")

    torch.save(model.cpu(), open(save_file, "wb"))

if __name__ == "__main__":
    root_dir_train = sys.argv[1]
    root_dir_test = sys.argv[2]
    embedding_file_train = sys.argv[3]
    embedding_file_test = sys.argv[4]
    seq_len_train = int(sys.argv[5])
    seq_len_test = int(sys.argv[6])
    epochs = int(sys.argv[7])
    save_file = sys.argv[8]

    main(root_dir_train, root_dir_test, embedding_file_train, embedding_file_test, seq_len_train, seq_len_test, epochs, save_file)
