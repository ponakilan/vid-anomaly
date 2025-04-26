import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from .dataset import TestFrameDataset
from .model import FramePredictor


def scores_to_labels(scores: list, threshold: int):
    return [int(score > threshold) for score in scores]


def get_best_accuracy(errors: list, labels: list):
    accuracies = {}
    for threshold in set(errors):
        predicted_labels = scores_to_labels(errors, threshold)
        accuracy = accuracy_score(labels, predicted_labels)
        accuracies[accuracy] = threshold
    best_accuracy = max(accuracies.keys())
    return best_accuracy, accuracies[best_accuracy]


def test_on_frames(
    model: FramePredictor,
    dataset: TestFrameDataset,
    rank: int
) -> float:

    errors = []
    labels = []
    model.eval()
    model.cuda(rank)
    loss_func = torch.nn.MSELoss()

    for i, (sequence, next_frame, anomaly) in tqdm(enumerate(dataset)):
        sequence, next_frame, anomaly = sequence.cuda(rank), next_frame.cuda(rank), anomaly.cuda(rank)

        with torch.no_grad():
            predicted_frame = model(sequence.unsqueeze(0))
        errors.append(loss_func(predicted_frame, next_frame.unsqueeze(0)).cpu().item())
        labels.append(anomaly.cpu().item())

    return get_best_accuracy(errors, labels)


def test_with_reconstruction(
        root_dir: str,
        labels_path: str,
        model: FramePredictor,
        seq_len: int,
        image_shape: tuple|list
):
    test_frame_dataset = TestFrameDataset(
        root_dir=root_dir,
        labels_dir=labels_path,
        seq_len=seq_len,
        image_shape=image_shape
    )

    frames_accuracy = test_on_frames(
        model=model,
        dataset=test_frame_dataset,
        rank=1 # Train on second GPU
    )
    return frames_accuracy