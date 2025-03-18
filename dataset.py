import os
import math

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, seq_len: int = 10, max_frames: int = 200, transform=None):
        self.root_dir = root_dir
        self.files = sorted(list(os.walk(self.root_dir)), key=lambda x: x[0])[1:]

        self.files = [list(directory) for directory in self.files]
        for i in range(len(self.files)):
            self.files[i][2] = sorted([file for file in self.files[i][2] if file[-3:] == "tif"])
            
        self.seq_len = seq_len
        self.max_frames = max_frames

    def __len__(self):
        return (len(self.files) * self.max_frames) // self.seq_len

    def _get_files(self, index: int):
        total_frames = index * self.seq_len + self.seq_len

        # Identify the directory
        dir_index = math.ceil(total_frames / self.max_frames) - 1

        # Start and end indices of files in the directory
        start = (index * self.seq_len) % self.max_frames
        end = start + self.seq_len

        return dir_index, start, end

    def __getitem__(self, index: int):
        directory, start, end = _get_files(index)


        return 1

dataset = ImageDataset("data/UCSDped1/Train", 10)
print(dataset._get_files(20))
