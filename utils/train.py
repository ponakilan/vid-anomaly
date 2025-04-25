import torch
from tqdm.auto import tqdm

def train_one_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        rank
):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    processed_batches = 0

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
