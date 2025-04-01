import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.models import FrameReconstructionModel
from core.dataset import ImageDataset, ImageEmbeddingDataset, EmbeddingDataset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dataset_train = ImageDataset(root_dir=args.dataset, seq_len=args.seqlen)
    embedding_dataset_train = EmbeddingDataset(embeddings_path=args.embeddings)
    dataset_train = ImageEmbeddingDataset(image_dataset=image_dataset_train, embedding_dataset=embedding_dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=12)

    model = FrameReconstructionModel(device=device).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_one_epoch():
        running_loss = 0.
        for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(dataloader_train)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        avg_loss = train_one_epoch()
        print(f"Epoch loss: {avg_loss:.6f}")

    torch.save(model.cpu(), open(args.ckpt, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a frame reconstruction model.")
    parser.add_argument("--dataset", type=str, required=True, help="Root directory for training images.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to the training embeddings file.")
    parser.add_argument("--seqlen", type=int, required=True, help="Sequence length for training.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--ckpt", type=str, required=True, help="File path to save the trained model.")
    
    args = parser.parse_args()
    main(args)
