import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = os.walk(self.root_dir)
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return 1


