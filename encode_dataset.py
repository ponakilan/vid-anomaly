import sys

import tqdm
import pickle
from dataset import EmbeddingGenerator


def generate_embeddings(root_dir: str, seq_len: int, save_location: str):
    dataset = EmbeddingGenerator(
        root_dir=root_dir,
        seq_len=seq_len
    )

    embeddings = []
    for i in tqdm.tqdm(range(len(dataset))):
        embeddings.append(dataset[i].cpu().tolist())

    with open(save_location, "wb") as f:
        pickle.dump(embeddings, f)
    
    print(f"{len(embeddings)} embeddings saved to {save_location}.")


if __name__ == "__main__":
    root_dir = sys.argv[1]
    seq_len = int(sys.argv[2])
    save_location = sys.argv[3]

    generate_embeddings(root_dir, seq_len, save_location)
