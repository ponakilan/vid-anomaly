import os
import math
import pickle

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import PIL
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


class EmbeddingDataset(Dataset):
    """
    Retrieves pre-computed embeddings from a pickle file.
    """
    def __init__(self, embeddings_path: str):
        with open(embeddings_path, "rb") as embeddings_file:
            self.embeddings = pickle.load(embeddings_file)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index: int):
        return torch.tensor(self.embeddings[index])


class ImageDataset(Dataset):
    def __init__(
            self, 
            root_dir: str, 
            seq_len: int = 10, 
            max_frames: int = 200
        ):
        self.root_dir = root_dir
        self.files = sorted(list(os.walk(self.root_dir)), key=lambda x: x[0])[1:]
        self.files = [directory for directory in self.files if "_gt" not in directory[0]]

        self.files = [list(directory) for directory in self.files]
        for i in range(len(self.files)):
            self.files[i][2] = sorted([file for file in self.files[i][2] if file[-3:] == "tif" or file[-3:] == "jpg"])

        self.seq_len = seq_len
        self.max_frames = max_frames

    def __len__(self):
        return (len(self.files) * self.max_frames) // self.seq_len

    def _get_files(self, index: int):
        total_frames = index * self.seq_len + self.seq_len
        dir_index = math.ceil(total_frames / self.max_frames) - 1

        start = (index * self.seq_len) % self.max_frames
        end = start + self.seq_len

        return dir_index, start, end
    
    def _load_images(
            self, 
            base_path: str, 
            files: str, 
            shape: tuple[int, int] = (224, 224)
        ):

        images = []
        for file in files:
            try:
                image = Image.open(f"{base_path}/{file}").convert('L').convert('RGB')
            except OSError:
                image = PIL.Image.new(mode = "RGB", size = (224, 224), color = (0, 0, 0))
            images.append(image)

        images_resized = [image.resize(shape) for image in images]
        return images_resized

    def __getitem__(self, index: int):
        directory, start, end = self._get_files(index)
        base_path = self.files[directory][0]
        files = self.files[directory][2][start:end]
        images = self._load_images(base_path, files)
        images_t = [transforms.functional.pil_to_tensor(image)/255.0 for image in images]
        return torch.stack(images_t)
    

class ImageEmbeddingDataset(Dataset):
    def __init__(
            self, 
            image_dataset: ImageDataset, 
            embedding_dataset: EmbeddingDataset
        ):
        self.image_dataset = image_dataset
        self.embedding_dataset = embedding_dataset

        assert len(self.image_dataset) == len(self.embedding_dataset)
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, index: int):
        images = self.image_dataset[index]
        embeddings = self.embedding_dataset[index]
        return (embeddings.float(), images.float())


class EmbeddingGenerator(Dataset):
    """
    Calculates embeddings for a sequence of frames using google/vit-base-patch16-224-in21k.
    """
    def __init__(
            self, 
            root_dir: str, 
            seq_len: int = 10, 
            max_frames: int = 200
        ):
        self.root_dir = root_dir
        self.files = sorted(list(os.walk(self.root_dir)), key=lambda x: x[0])[1:]
        self.files = [directory for directory in self.files if "_gt" not in directory[0]]

        self.files = [list(directory) for directory in self.files]
        for i in range(len(self.files)):
            self.files[i][2] = sorted([file for file in self.files[i][2] if file[-3:] == "tif" or file[-3:] == "jpg"])

        self.seq_len = seq_len
        self.max_frames = max_frames
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)

    def __len__(self):
        return (len(self.files) * self.max_frames) // self.seq_len

    def _get_files(self, index: int):
        total_frames = index * self.seq_len + self.seq_len
        dir_index = math.ceil(total_frames / self.max_frames) - 1

        start = (index * self.seq_len) % self.max_frames
        end = start + self.seq_len

        return dir_index, start, end

    def _load_images(
            self, 
            base_path: str, 
            files: str, 
            shape: tuple[int, int] = (224, 224)
        ):
        
        images = []
        for file in files:
            try:
                image = PIL.Image.new(mode = "RGB", size = (224, 224), color = (0, 0, 0))
            except OSError:
                image = torch.zeros(3, 224, 224)
            images.append(image)

        images_resized = [image.resize(shape) for image in images]
        return images_resized

    def _load_frames(self, base_path: str, files: str):
        inputs = self.processor(
            images=self._load_images(base_path, files), 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        encoding = last_hidden_states[:, 0, :]
        return encoding

    def __getitem__(self, index: int):
        directory, start, end = self._get_files(index)
        base_path = self.files[directory][0]
        files = self.files[directory][2][start:end]
        return self._load_frames(base_path, files)
