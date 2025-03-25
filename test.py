import sys
import torch
from tqdm import tqdm
import pandas as pd

from core.dataset import EmbeddingDataset, ImageDataset, ImageEmbeddingDataset
from core.models import FrameReconstructionModel

train_dir = "data/UCSDped1/Train"
test_dir = "data/UCSDped1/Test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_path, embeddings_path, output_file):
    embedding_dataset_test = EmbeddingDataset(
        embeddings_path=embeddings_path
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

    model = torch.load(open(model_path, "rb"), weights_only=False).to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()

    test_errors = []
    indices = []
    for i in range(1, len(test_dataloader)//4 + 1):
        indices.extend([i, i, i, i])

    for test_embeddings, test_images in tqdm(test_dataloader):
        test_embeddings, test_images = test_embeddings.to(device), test_images.to(device)
        out = model(test_embeddings)
        error = loss_fn(out, test_images)
        test_errors.append(error.item())

    errors_df = pd.DataFrame({
        "test_error": test_errors,
        "sequence_number": indices 
    })
    errors_df.to_csv(output_file)

if __name__ == "__main__":
    args = sys.argv
    model_path = args[1]
    embeddings_path = args[2]
    output_file = args[3]
    main(model_path, embeddings_path, output_file)
    