import sys
import torch
from tqdm import tqdm
import pandas as pd

from dataset import EmbeddingGenerator, ImageDataset, ImageEmbeddingDataset
from models import FrameReconstructionModel

train_dir = "data/UCSDped1/Train"
test_dir = "data/UCSDped1/Test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dataset_train = EmbeddingGenerator(
    root_dir=train_dir,
    seq_len=50
)
image_dataset_train = ImageDataset(
    root_dir=train_dir,
    seq_len=50
)
train_dataset = ImageEmbeddingDataset(
    image_dataset=image_dataset_train,
    embedding_dataset=embedding_dataset_train
)
train_dataloader = torch.utils.data.DataLoader(train_dataset)

embedding_dataset_test = EmbeddingGenerator(
    root_dir=test_dir,
    seq_len=50
)
image_dataset_test = ImageDataset(
    root_dir=test_dir,
    seq_len=50
)
test_dataset = ImageEmbeddingDataset(
    image_dataset=image_dataset_test,
    embedding_dataset=embedding_dataset_test
)
test_dataloader = torch.utils.data.DataLoader(test_dataset)

def main(model_path):
    model = torch.load(open(model_path, "rb"), weights_only=False).to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()

    train_errors = []
    test_errors = []

    for train_embeddings, train_images in tqdm(train_dataloader):
        train_embeddings, train_images = train_embeddings.to(device), train_images.to(device)
        out = model(train_embeddings)
        error = loss_fn(out, train_images)
        train_errors.append(error.item())

    # for test_embeddings, test_images in tqdm(test_dataloader):
    #     test_embeddings, test_images = test_embeddings.to(device), test_images.to(device)
    #     out = model(test_embeddings)
    #     error = loss_fn(out, test_images)
    #     test_errors.append(error.item())

    errors_df = pd.DataFrame({
        "train_error": train_errors,
        # "test_error": test_errors
    })
    errors_df.to_csv("errors.csv")

if __name__ == "__main__":
    args = sys.argv
    model_path = args[1]
    main(model_path)
    