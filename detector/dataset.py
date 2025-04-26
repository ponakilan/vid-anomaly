import os

import torch
import numpy as np
from PIL import Image
from requests.packages import target
from torchvision import transforms
from torch.utils.data import Dataset

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class SequenceDataset(Dataset):
    def __init__(self, root_dir: str, seq_len: int, image_shape: tuple):
        self.seq_len = seq_len
        self.image_tensor_shape = image_shape
        self.sequences = self.load_sequences(root_dir=root_dir)
        print(f"{len(self.sequences)} loaded")

        self.cumulative_lengths = [0]
        for sequence in self.sequences:
            valid_indices = max(0, sequence.shape[0] - seq_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_indices)

    def load_one_sequence(self, sequence_dir: str):
        frame_files = [file for file in os.listdir(sequence_dir) if file.endswith("jpg")]
        sequence = torch.zeros(len(frame_files), *self.image_tensor_shape)
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(sequence_dir, frame_file)
            image = Image.open(frame_path).convert("RGB")
            sequence[i] = TRANSFORM(image)
        return sequence

    def load_sequences(self, root_dir: str):
        sequences = []
        sequence_dirs = os.listdir(root_dir)
        for sequence_dir in sequence_dirs:
            sequence_dir_path = os.path.join(root_dir, sequence_dir)
            sequence = self.load_one_sequence(sequence_dir_path)
            sequences.append(sequence)
        return sequences

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_idx = 0
        while idx >= self.cumulative_lengths[tensor_idx + 1]:
            tensor_idx += 1

        start_frame = idx - self.cumulative_lengths[tensor_idx]
        tensor = self.sequences[tensor_idx]
        input_sequence = tensor[start_frame:start_frame + self.seq_len]
        target_frame = tensor[start_frame + self.seq_len]

        return input_sequence, target_frame


class TestFrameDataset(Dataset):
    def __init__(self, root_dir: str, labels_dir: str, seq_len: int, image_shape: tuple):
        self.seq_len = seq_len
        self.image_tensor_shape = image_shape
        self.labels_dir = labels_dir
        self.label_files = os.listdir(labels_dir)
        self.sequences = self.load_sequences(root_dir=root_dir)
        print(f"{len(self.sequences)} sequences loaded")

        self.cumulative_lengths = [0]
        for sequence in self.sequences:
            valid_indices = max(0, sequence.shape[0] - seq_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_indices)

    def load_one_sequence(self, sequence_dir: str):
        frame_files = [file for file in os.listdir(sequence_dir) if file.endswith("jpg")]
        sequence = torch.zeros(len(frame_files), *self.image_tensor_shape)
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(sequence_dir, frame_file)
            image = Image.open(frame_path).convert("RGB")
            sequence[i] = TRANSFORM(image)
        return sequence

    def load_sequences(self, root_dir: str):
        sequences = []
        sequence_dirs = os.listdir(root_dir)
        for sequence_dir in sequence_dirs:
            sequence_dir_path = os.path.join(root_dir, sequence_dir)
            sequence = self.load_one_sequence(sequence_dir_path)
            sequences.append(sequence)
        return sequences

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_idx = 0
        while idx >= self.cumulative_lengths[tensor_idx + 1]:
            tensor_idx += 1

        start_frame = idx - self.cumulative_lengths[tensor_idx]
        tensor = self.sequences[tensor_idx]
        label_file = self.label_files[tensor_idx]
        label_file_path = os.path.join(self.labels_dir, label_file)
        labels = np.load(label_file_path).tolist()

        input_sequence = tensor[start_frame:start_frame + self.seq_len]
        target_frame = tensor[start_frame + self.seq_len]
        target_label = labels[start_frame + self.seq_len]

        return input_sequence, target_frame, torch.tensor(int(target_label))
