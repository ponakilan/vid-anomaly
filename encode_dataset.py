import tqdm
import pickle
from dataset import ImageDataset

dataset = ImageDataset(
    root_dir="data/UCSDped1/Train",
    seq_len=100
)

embeddings = []
for i in tqdm.tqdm(range(len(dataset))):
    embeddings.append(dataset[i].cpu().tolist())

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
