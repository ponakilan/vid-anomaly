import os
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from tqdm.auto import tqdm
import argparse


parser = argparse.ArgumentParser(description='Frame Prediction with Multi-GPU Training')
parser.add_argument('--num_gpus', type=int, default=None,
                    help='Number of GPUs to use (default: use all available GPUs)')
parser.add_argument('--gpu_ids', type=str, default=None,
                    help='Specific GPU IDs to use, comma-separated (e.g., "0,2,3")')
parser.add_argument('--batch_size', type=int, default=12,
                    help='Batch size per GPU')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')

args = parser.parse_args()

configs = {
    "ROOT_DIR": "data/IPAD/R01/training/frames",
    "IMAGE_TENSOR_SHAPE": (3, 224, 224),
    "MAE_BACKBONE": "OpenGVLab/VideoMAEv2-Large",
    "SEQ_LEN": 16,
    "BATCH_SIZE": args.batch_size,  # Per GPU batch size
    "DATASET_SHUFFLE": True,
    "EPOCHS": args.epochs,
    "LEARNING_RATE": 1e-3
}

# Determine which GPUs to use
available_gpus = torch.cuda.device_count()

if args.gpu_ids is not None:
    # Use specific GPU IDs provided by user
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    world_size = len(gpu_ids)
elif args.num_gpus is not None:
    # Use the specified number of GPUs
    world_size = min(args.num_gpus, available_gpus)
    # By default, we'll use the first N GPUs
    gpu_ids = list(range(world_size))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
else:
    # Use all available GPUs
    world_size = available_gpus
    gpu_ids = list(range(world_size))

TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])


class CNNFrameReconstructor(torch.nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=512, out_channels=3, img_size=224):
        super(CNNFrameReconstructor, self).__init__()
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.out_channels = out_channels

        self.fc = torch.nn.Linear(embed_dim, feature_dim * (img_size // 16) * (img_size // 16))

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(feature_dim // 4, feature_dim // 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(feature_dim // 8, out_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, embed_dim) - Single embedding vector per batch item
        Output: (B, C, H, W) - Single frame per batch item
        """
        B, D = x.shape
        x = self.fc(x)
        x = x.view(B, self.feature_dim, self.img_size // 16, self.img_size // 16)
        x = self.decoder(x)
        return x


class FramePredictor(torch.nn.Module):
    _reconstruct: bool

    def __init__(self):
        super(FramePredictor, self).__init__()

        config = AutoConfig.from_pretrained(configs["MAE_BACKBONE"], trust_remote_code=True)
        self.processor = VideoMAEImageProcessor.from_pretrained(configs["MAE_BACKBONE"])
        self.video_mae = AutoModel.from_pretrained(configs["MAE_BACKBONE"], config=config, trust_remote_code=True)
        self._reconstruct = True

        self.reconstructor = CNNFrameReconstructor()

    def set_reconstruct(self, val):
        self._reconstruct = val

    def forward(self, x):
        videos = [list(sequence) for sequence in x]
        processed = self.processor(videos, return_tensors="pt")
        processed['pixel_values'] = processed['pixel_values'].permute(0, 2, 1, 3, 4).to(x.device)
        output = self.video_mae(**processed)
        if self._reconstruct:
            output = self.reconstructor(output)
        return output


def load_one_sequence(sequence_dir: str):
    frame_files = [file for file in os.listdir(sequence_dir) if file.endswith("jpg")]
    sequence = torch.zeros(len(frame_files), *configs["IMAGE_TENSOR_SHAPE"])
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(sequence_dir, frame_file)
        image = Image.open(frame_path).convert("RGB")
        sequence[i] = TRANSFORM(image)
    return sequence


def load_sequences(root_dir: str):
    sequences = []
    sequence_dirs = os.listdir(root_dir)
    for sequence_dir in sequence_dirs:
        sequence_dir_path = os.path.join(root_dir, sequence_dir)
        sequence = load_one_sequence(sequence_dir_path)
        sequences.append(sequence)
    return sequences


def visualize_sequence(sequence: torch.Tensor):
    """
    Expected input shape: (T, C, H, W)
    """
    images = sequence.permute(0, 2, 3, 1)
    grid_shape = math.ceil(math.sqrt(images.shape[0]))
    for i, image in enumerate(images):
        plt.subplot(grid_shape, grid_shape, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


class SequenceDataset(Dataset):
    def __init__(self, tensors, seq_len=10):
        self.tensors = tensors
        self.seq_len = seq_len

        self.cumulative_lengths = [0]
        for tensor in tensors:
            valid_indices = max(0, tensor.shape[0] - seq_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_indices)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        tensor_idx = 0
        while idx >= self.cumulative_lengths[tensor_idx + 1]:
            tensor_idx += 1

        start_frame = idx - self.cumulative_lengths[tensor_idx]
        tensor = self.tensors[tensor_idx]
        input_sequence = tensor[start_frame:start_frame + self.seq_len]
        target_frame = tensor[start_frame + self.seq_len]

        return input_sequence, target_frame


def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()


def train_one_epoch(model, train_loader, optimizer, criterion, rank):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    processed_batches = 0

    # Only show progress bar on the main process
    train_iter = tqdm(train_loader, desc=f"Training on GPU {rank}", disable=rank != 0)

    for batch_idx, (sequence, target) in enumerate(train_iter):
        sequence, target = sequence.cuda(rank), target.cuda(rank)

        optimizer.zero_grad()
        output = model(sequence)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1

        if rank == 0 and batch_idx % 10 == 0:
            train_iter.set_postfix(loss=total_loss / processed_batches)

    return total_loss / processed_batches


def validate(model, val_loader, criterion, rank):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    processed_batches = 0

    # Only show progress bar on the main process
    val_iter = tqdm(val_loader, desc=f"Validating on GPU {rank}", disable=rank != 0)

    with torch.no_grad():
        for batch_idx, (sequence, target) in enumerate(val_iter):
            sequence, target = sequence.cuda(rank), target.cuda(rank)

            output = model(sequence)
            loss = criterion(output, target)

            total_loss += loss.item()
            processed_batches += 1

            if rank == 0 and batch_idx % 10 == 0:
                val_iter.set_postfix(loss=total_loss / processed_batches)

    return total_loss / processed_batches


def run_training(rank, world_size, sequences):
    """Run training on a single process"""
    # Setup distributed environment
    setup(rank, world_size)

    # Create dataset
    dataset = SequenceDataset(
        tensors=sequences,
        seq_len=configs["SEQ_LEN"]
    )

    # Split into train and validation sets
    generator = torch.Generator().manual_seed(0)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, valid_ds = random_split(dataset, [train_len, val_len], generator=generator)

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=configs["DATASET_SHUFFLE"]
    )

    val_sampler = DistributedSampler(
        valid_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=configs["BATCH_SIZE"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        valid_ds,
        batch_size=configs["BATCH_SIZE"],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    model = FramePredictor().cuda(rank)

    model = DDP(model, device_ids=[rank])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["LEARNING_RATE"])

    best_val_loss = float('inf')

    if rank == 0:
        print(f"Starting training for {configs['EPOCHS']} epochs on {world_size} GPUs")

    for epoch in range(configs["EPOCHS"]):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # Train and validate
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, rank)
        val_loss = validate(model, val_loader, criterion, rank)

        # Print statistics from main process
        if rank == 0:
            print(f"Epoch {epoch + 1}/{configs['EPOCHS']}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model weights only from the main process
                torch.save(model.module.state_dict(), "models/IPAD_R01.pth")
                print(f"New best model saved with validation loss: {best_val_loss:.6f}")

    # Clean up
    cleanup()


def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return

    print(f"Loading sequences from {configs['ROOT_DIR']}...")
    sequences = load_sequences(configs["ROOT_DIR"])
    print(f"Number of sequences loaded: {len(sequences)}")

    if world_size == 0:
        print("No GPUs available or selected. Exiting.")
        return

    # Create processes for each GPU
    print(f"Using {world_size} GPUs: {gpu_ids}")
    mp.spawn(
        run_training,
        args=(world_size, sequences),
        nprocs=world_size,
        join=True
    )

    print("Training complete!")

    if os.path.exists("models/IPAD_R01.pth"):
        print("Loading best model for inference...")
        model = FramePredictor().cuda()
        model.load_state_dict(torch.load("models/IPAD_R01.pth"))
        print("Model loaded successfully!")


if __name__ == "__main__":
    # This protects the main function from being executed when spawned processes import this file
    main()