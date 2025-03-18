import os
import math

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import ViTImageProcessor, ViTModel

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, seq_len: int = 10, max_frames: int = 200):
        self.root_dir = root_dir
        self.files = sorted(list(os.walk(self.root_dir)), key=lambda x: x[0])[1:]

        self.files = [list(directory) for directory in self.files]
        for i in range(len(self.files)):
            self.files[i][2] = sorted([file for file in self.files[i][2] if file[-3:] == "tif"])

        self.seq_len = seq_len
        self.max_frames = max_frames

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def __len__(self):
        return (len(self.files) * self.max_frames) // self.seq_len

    def _get_files(self, index: int):
        total_frames = index * self.seq_len + self.seq_len
        dir_index = math.ceil(total_frames / self.max_frames) - 1

        start = (index * self.seq_len) % self.max_frames
        end = start + self.seq_len

        return dir_index, start, end

    def _load_frames(self, base_path: str, files: str):
        inputs = self.processor(
            images=[Image.open(f"{base_path}/{file}").convert('L').convert('RGB') for file in files], 
            return_tensors="pt"
        )
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        encoding = last_hidden_states[:, 0, :]
        return encoding

    def __getitem__(self, index: int):
        directory, start, end = self._get_files(index)
        base_path = self.files[directory][0]
        files = self.files[directory][2][start:end]
        return self._load_frames(base_path, files)
