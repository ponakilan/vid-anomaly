import os
import yaml
import logging
import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.general import validate
from utils.ddp import setup, cleanup
from detector.model import FramePredictor
from detector.train import train_one_epoch
from detector.dataset import SequenceDataset
from detector.test import test_with_reconstruction

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open("train_config.yaml", "r") as config_file:
    configs = yaml.safe_load(config_file)

parser = argparse.ArgumentParser(description='Frame Prediction with Multi-GPU Training')
parser.add_argument('--num_gpus', type=int, default=None,
                    help='Number of GPUs to use (default: use all available GPUs)')
parser.add_argument('--gpu_ids', type=str, default=None,
                    help='Specific GPU IDs to use, comma-separated (e.g., "0,2,3")')

args = parser.parse_args()

available_gpus = torch.cuda.device_count()

if args.gpu_ids is not None:
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    world_size = len(gpu_ids)

elif args.num_gpus is not None:
    world_size = min(args.num_gpus, available_gpus)
    gpu_ids = list(range(world_size))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))

else:
    world_size = available_gpus
    gpu_ids = list(range(world_size))


def run_training(rank, wrld_size, train_ds, valid_ds):
    """Run training on a single process"""
    logger.info(f"Setting up distributed environment for GPU {rank}")
    setup(rank, wrld_size)

    if rank == 0:
        tb_writer = SummaryWriter(log_dir=os.path.join('runs', configs.get('RUN_NAME', 'frame_predictor')))
        config_text = yaml.dump(configs, default_flow_style=False)
        tb_writer.add_text("Configuration", f"```yaml\n{config_text}\n```", global_step=0)

    logger.info(f"Creating dataloaders for GPU {rank}")
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=wrld_size,
        rank=rank,
        shuffle=configs["DATASET_SHUFFLE"]
    )

    val_sampler = DistributedSampler(
        valid_ds,
        num_replicas=wrld_size,
        rank=rank,
        shuffle=False
    )

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

    logger.info(f"Preparing model for distributed training in GPU {rank}")
    model = FramePredictor(mae_backbone=configs["MAE_BACKBONE"])

    if os.path.exists(configs["CHECKPOINT_NAME"]):
        logger.info(f"Resuming training from checkpoint: {configs['CHECKPOINT_NAME']}")
        model.load_state_dict(torch.load(configs["CHECKPOINT_NAME"]))
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["LEARNING_RATE"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_loss = float('inf')

    if rank == 0:
        logger.info(f"Starting training for {configs['EPOCHS']} epochs on {wrld_size} GPUs")

    for epoch in range(configs["EPOCHS"]):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, rank)
        val_loss = validate(model, val_loader, criterion, rank)
        test_accuracy = test_with_reconstruction(
            root_dir=configs["TEST_ROOT_DIR"],
            labels_path=configs["TEST_LABELS"],
            model=model,
            seq_len=configs["SEQ_LEN"],
            image_shape=configs["IMAGE_TENSOR_SHAPE"]
        )

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']

            tb_writer.add_scalar('Loss/train', train_loss, epoch)
            tb_writer.add_scalar('Loss/validation', val_loss, epoch)
            tb_writer.add_scalar('Accuracy', test_accuracy, epoch)
            tb_writer.add_scalar('Learning_rate', current_lr, epoch)

            print(f"Epoch {epoch + 1}/{configs['EPOCHS']}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Accuracy: {test_accuracy:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), configs["CHECKPOINT_NAME"])
                print(f"New best model saved with validation loss: {best_val_loss:.6f}")
                tb_writer.add_scalar('Best_val_loss', best_val_loss, epoch)

        lr_scheduler.step(val_loss)

    if rank == 0:
        tb_writer.close()

    cleanup()


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return

    if world_size == 0:
        print("No GPUs available or selected. Exiting.")
        return

    logger.info("Loading the dataset")
    dataset = SequenceDataset(
        root_dir=configs["ROOT_DIR"],
        seq_len=configs["SEQ_LEN"],
        image_shape=configs["IMAGE_TENSOR_SHAPE"]
    )

    generator = torch.Generator().manual_seed(0)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, valid_ds = random_split(dataset, [train_len, val_len], generator=generator)
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(valid_ds)}")

    logger.info(f"Using {world_size} GPUs: {gpu_ids}")
    mp.spawn(
        run_training,
        args=(world_size, train_ds, valid_ds),
        nprocs=world_size,
        join=True
    )

    print("Training complete!")


if __name__ == "__main__":
    main()