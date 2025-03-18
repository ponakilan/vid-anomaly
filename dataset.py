import os
import math

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, seq_len: int = 10, max_frames: int = 200, transform=None):
        self.root_dir = root_dir
        self.files = sorted(list(os.walk(self.root_dir)), key=lambda x: x[0])[1:]

        self.files = [list(directory) for directory in self.files]
        for i in range(len(self.files)):
            self.files[i][2] = sorted([file for file in self.files[i][2] if file[-3:] == "tif"])

        self.seq_len = seq_len
        self.max_frames = max_frames
        self.transform = transform

    def __len__(self):
        return (len(self.files) * self.max_frames) // self.seq_len

    def _get_files(self, index: int):
        total_frames = index * self.seq_len + self.seq_len
        dir_index = math.ceil(total_frames / self.max_frames) - 1

        start = (index * self.seq_len) % self.max_frames
        end = start + self.seq_len

        return dir_index, start, end

    def _load_image(self, img_path: str):
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def _load_frames(self, base_path: str, files: str):
        return torch.stack([self._load_image(f'{base_path}/{img_name}') for img_name in files])

    def __getitem__(self, index: int):
        directory, start, end = self._get_files(index)
        base_path = self.files[directory][0]
        files = self.files[directory][2][start:end]
        return self._load_frames(base_path, files)
        