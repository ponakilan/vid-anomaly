import math

import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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