import os
import logging # Optional: for better error messages
from typing import Iterable, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Setup basic logging (optional but helpful)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- SequenceDataset remains the same ---
class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    # ... (keep the original SequenceDataset code) ...
    def __init__(self, root_dir: str, seq_len: int, image_shape: List[int]):
        self.seq_len = seq_len
        self.image_tensor_shape = image_shape
        # Load sequences directly (no labels needed here)
        raw_sequences = self._load_all_sequences(root_dir)
        self.sequences = [seq for seq, _ in raw_sequences] # Keep only tensors
        log.info(f"{len(self.sequences)} sequences loaded for SequenceDataset")

        self.cumulative_lengths = [0]
        for sequence in self.sequences:
            if sequence.shape[0] > seq_len:
                 valid_indices = sequence.shape[0] - seq_len
                 self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_indices)

    def _load_one_sequence(self, sequence_dir: str) -> Optional[torch.Tensor]:
        frame_files = sorted([f for f in os.listdir(sequence_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))])
        if not frame_files:
            log.warning(f"No image files found in {sequence_dir}. Skipping.")
            return None

        sequence = torch.zeros(len(frame_files), *self.image_tensor_shape)
        try:
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(sequence_dir, frame_file)
                with Image.open(frame_path) as img:
                    image = img.convert("RGB")
                sequence[i] = TRANSFORM(image) # type: ignore
            return sequence
        except Exception as e:
            log.error(f"Error loading sequence from {sequence_dir}: {e}")
            return None

    def _load_all_sequences(self, root_dir: str) -> List[Tuple[torch.Tensor, str]]:
        sequences_data = []
        if not os.path.isdir(root_dir):
            log.error(f"Root directory not found: {root_dir}")
            return []

        sequence_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for sequence_dir_name in sequence_dirs:
            sequence_dir_path = os.path.join(root_dir, sequence_dir_name)
            sequence = self._load_one_sequence(sequence_dir_path)
            if sequence is not None:
                 # Store sequence tensor and its original directory name for reference
                sequences_data.append((sequence, sequence_dir_name))
        return sequences_data

    def __len__(self):
        # Handle case where no valid sequences were loaded
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
             raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")

        # Find the sequence this index belongs to
        tensor_idx = 0
        # Corrected loop condition for safety
        while tensor_idx < len(self.cumulative_lengths) - 1 and idx >= self.cumulative_lengths[tensor_idx + 1]:
            tensor_idx += 1

        # Check if tensor_idx is valid (safeguard against empty sequences list)
        if tensor_idx >= len(self.sequences):
             raise IndexError(f"Calculated tensor_idx {tensor_idx} is out of bounds for sequences list.")

        start_frame = idx - self.cumulative_lengths[tensor_idx]
        tensor = self.sequences[tensor_idx]

        # Double check bounds before slicing
        if start_frame < 0 or start_frame + self.seq_len >= tensor.shape[0]:
             raise IndexError(f"Calculated start_frame {start_frame} or end frame {start_frame + self.seq_len} is out of bounds for tensor with shape {tensor.shape}")


        input_sequence = tensor[start_frame : start_frame + self.seq_len]
        target_frame = tensor[start_frame + self.seq_len]

        return input_sequence, target_frame


# --- Revised TestFrameDataset ---
class TestFrameDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, root_dir: str, labels_dir: str, seq_len: int, image_shape: List[int]):
        self.seq_len = seq_len
        self.image_tensor_shape = image_shape
        self.labels_dir = labels_dir

        # Load sequences and explicitly find corresponding label files
        self.sequences_and_labels = self._load_sequences_and_find_labels(root_dir, labels_dir)
        log.info(f"{len(self.sequences_and_labels)} sequence/label pairs loaded for TestFrameDataset")

        self.cumulative_lengths = [0]
        for sequence, label_path in self.sequences_and_labels:
            # Ensure sequence has enough frames for at least one item
            if sequence.shape[0] > seq_len:
                valid_indices = sequence.shape[0] - seq_len
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_indices)
            # else: # Optional: Log sequences that are too short
                # log.warning(f"Sequence (from {label_path}) with {sequence.shape[0]} frames is too short for seq_len {seq_len} and will be skipped.")

    def _load_one_sequence(self, sequence_dir: str) -> Optional[torch.Tensor]:
         # Same implementation as in SequenceDataset
        frame_files = sorted([f for f in os.listdir(sequence_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))])
        if not frame_files:
            log.warning(f"No image files found in {sequence_dir}. Skipping.")
            return None

        sequence = torch.zeros(len(frame_files), *self.image_tensor_shape)
        try:
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(sequence_dir, frame_file)
                with Image.open(frame_path) as img:
                    image = img.convert("RGB")
                sequence[i] = TRANSFORM(image) # type: ignore
            return sequence
        except Exception as e:
            log.error(f"Error loading sequence from {sequence_dir}: {e}")
            return None

    def _load_sequences_and_find_labels(self, root_dir: str, labels_dir: str) -> List[Tuple[torch.Tensor, str]]:
        sequences_data = []
        if not os.path.isdir(root_dir):
            log.error(f"Root directory not found: {root_dir}")
            return []
        if not os.path.isdir(labels_dir):
            log.error(f"Labels directory not found: {labels_dir}")
            return []

        # Get potential sequence directories
        sequence_dir_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for seq_dir_name in sequence_dir_names:
            # --- ASSUMPTION: Label file has the same base name as the directory, but with .npy extension ---
            # --- Adjust this logic if your naming convention is different ---
            expected_label_file = f"0{seq_dir_name}.npy"
            label_file_path = os.path.join(labels_dir, expected_label_file)

            sequence_dir_path = os.path.join(root_dir, seq_dir_name)

            if os.path.exists(label_file_path):
                sequence = self._load_one_sequence(sequence_dir_path)
                if sequence is not None:
                    # Optionally, load labels here once to check length immediately
                    try:
                        labels_check = np.load(label_file_path)
                        if len(labels_check) == sequence.shape[0]:
                            sequences_data.append((sequence, label_file_path))
                        else:
                           log.warning(f"Skipping sequence {seq_dir_name}: Frame count ({sequence.shape[0]}) != label count ({len(labels_check)}) in {label_file_path}")
                    except Exception as e:
                        log.error(f"Error loading or checking label file {label_file_path}: {e}. Skipping sequence.")
                # else: sequence loading failed (already logged)
            else:
                log.warning(f"Label file not found for sequence directory {seq_dir_name} (expected: {label_file_path}). Skipping.")

        return sequences_data

    def __len__(self):
         # Handle case where no valid sequences were loaded
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0


    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
             raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")


        # Find the sequence this index belongs to
        pair_idx = 0
        # Corrected loop condition for safety
        while pair_idx < len(self.cumulative_lengths) - 1 and idx >= self.cumulative_lengths[pair_idx + 1]:
            pair_idx += 1

        # Check if pair_idx is valid
        if pair_idx >= len(self.sequences_and_labels):
             raise IndexError(f"Calculated pair_idx {pair_idx} is out of bounds for sequences/labels list.")


        start_frame = idx - self.cumulative_lengths[pair_idx]
        tensor, label_file_path = self.sequences_and_labels[pair_idx] # Get sequence and its specific label path

        # Load the labels for this specific sequence
        try:
            labels = np.load(label_file_path).tolist()
        except Exception as e:
            log.error(f"Failed to load labels from {label_file_path} in __getitem__ for index {idx}: {e}")
            # Decide how to handle this: raise error, return None, return dummy data?
            raise RuntimeError(f"Failed to load labels from {label_file_path}") from e


        # --- The crucial check is now more reliable ---
        # Although we added a check during loading, it's good practice
        # to keep an assert here during development/debugging.
        # In production, you might remove it if the loading check is sufficient.
        assert len(labels) == tensor.shape[0], \
            f"Mismatch after loading! Frames: {tensor.shape[0]}, Labels: {len(labels)} in {label_file_path}"

        # Double check bounds before slicing
        if start_frame < 0 or start_frame + self.seq_len >= tensor.shape[0]:
             raise IndexError(f"Calculated start_frame {start_frame} or end frame {start_frame + self.seq_len + 1} is out of bounds for tensor with shape {tensor.shape}")
        if start_frame + self.seq_len >= len(labels):
             raise IndexError(f"Calculated target label index {start_frame + self.seq_len} is out of bounds for labels with length {len(labels)}")


        input_sequence = tensor[start_frame : start_frame + self.seq_len]
        target_frame = tensor[start_frame + self.seq_len]
        target_label = labels[start_frame + self.seq_len] # Get label for the target frame

        # Ensure target_label is compatible with torch.tensor
        try:
            target_label_tensor = torch.tensor(int(target_label))
        except ValueError:
             log.error(f"Could not convert target label '{target_label}' to int for index {idx}, file {label_file_path}")
             # Handle appropriately - maybe raise error or return a default?
             raise ValueError(f"Invalid label format: {target_label}")


        return input_sequence, target_frame, target_label_tensor

# --- Revised TestEmbeddingsDataset (apply similar logic) ---
class TestEmbeddingsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root_dir: str, labels_dir: str, seq_len: int, image_shape: List[int]):
        self.seq_len = seq_len
        self.image_tensor_shape = image_shape
        # Use the same loading logic as TestFrameDataset
        self._loader = TestFrameDataset(root_dir, labels_dir, seq_len, image_shape)
        log.info(f"{len(self._loader.sequences_and_labels)} sequence/label pairs loaded for TestEmbeddingsDataset")


    def __len__(self):
        # Delegate length calculation
        return len(self._loader)

    def __getitem__(self, idx: int):
        # Reuse the logic from TestFrameDataset to find the correct sequence and labels
        if idx < 0 or idx >= len(self):
             raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")

        pair_idx = 0
        while pair_idx < len(self._loader.cumulative_lengths) - 1 and idx >= self._loader.cumulative_lengths[pair_idx + 1]:
            pair_idx += 1

        if pair_idx >= len(self._loader.sequences_and_labels):
             raise IndexError(f"Calculated pair_idx {pair_idx} is out of bounds.")

        start_frame = idx - self._loader.cumulative_lengths[pair_idx]
        tensor, label_file_path = self._loader.sequences_and_labels[pair_idx]

        try:
            labels = np.load(label_file_path).tolist()
        except Exception as e:
            log.error(f"Failed to load labels from {label_file_path} in __getitem__ for index {idx}: {e}")
            raise RuntimeError(f"Failed to load labels from {label_file_path}") from e

        assert len(labels) == tensor.shape[0], \
            f"Mismatch after loading! Frames: {tensor.shape[0]}, Labels: {len(labels)} in {label_file_path}"

        # Double check bounds before slicing input sequence
        if start_frame < 0 or start_frame + self.seq_len > tensor.shape[0]: # Note: > not >= because slice is exclusive at end
             raise IndexError(f"Calculated start_frame {start_frame} or end frame {start_frame + self.seq_len} is out of bounds for tensor with shape {tensor.shape}")

        input_sequence = tensor[start_frame : start_frame + self.seq_len]

        # --- Calculate target_label based on the INPUT sequence's labels ---
        # Ensure the range for checking labels is within the bounds of the loaded labels list
        label_check_end = min(start_frame + self.seq_len, len(labels))
        if start_frame < 0 or start_frame >= len(labels):
             raise IndexError(f"start_frame {start_frame} out of bounds for labels list (len {len(labels)})")

        target_label = any(
            bool(labels[i]) for i in range(start_frame, label_check_end)
        )

        return input_sequence, torch.tensor(int(target_label))


