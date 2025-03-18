from dataset import ImageDataset

dataset = ImageDataset(root_dir="data/UCSDped1/Train")
print(dataset[12].shape)

import tqdm
import pickle

embeddings = []
for i in tqdm.tqdm(range(len(dataset))):
    embeddings.append(dataset[i].cpu().tolist())

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
