#!/usr/bin/env python3

import argparse
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.models import FrameReconstructionModel
from core.dataset import ImageDataset, ImageEmbeddingDataset, EmbeddingDataset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FrameReconstructionModel(device=device).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_one_epoch(dataloader):
        running_loss = 0.
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(dataloader)

    for epoch in range(args["epochs"]):
        print(f"Epoch {epoch+1}/{args['epochs']}")
        for dataset in args["datasets"]:
            print(f"Training on dataset: {dataset['root_dir_train']}")
            image_dataset_train = ImageDataset(root_dir=dataset["root_dir_train"], seq_len=dataset["seq_len_train"])
            embedding_dataset_train = EmbeddingDataset(embeddings_path=dataset["embedding_file_train"])
            dataset_train = ImageEmbeddingDataset(image_dataset=image_dataset_train, embedding_dataset=embedding_dataset_train)
            dataloader_train = DataLoader(dataset_train, batch_size=dataset["batch_size"])
            avg_loss = train_one_epoch(dataloader_train)
            print(f"Dataset loss: {avg_loss:.6f}")

    torch.save(model.cpu(), open(args["save_file"], "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a frame reconstruction model using a JSON config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    main(config)
