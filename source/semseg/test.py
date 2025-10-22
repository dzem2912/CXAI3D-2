import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score,
                             mutual_info_score)

from model import PointNet2SemSegMsg
from dataset import PointCloudInstancesDataset, get_train_val_file_paths

def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str):
    if metric_name == 'balanced_accuracy_score':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric_name == 'accuracy_score':
        return accuracy_score(y_true, y_pred, average='weighted')
    elif metric_name == 'f1_score':
        return f1_score(y_true, y_pred, average='weighted')
    elif metric_name == 'precision_score':
        return precision_score(y_true, y_pred, average='weighted')
    elif metric_name == 'recall_score':
        return recall_score(y_true, y_pred, average='weighted')
    elif metric_name == 'mutual_info_score':
        return mutual_info_score(y_true, y_pred)
    else:
        raise ValueError(f'Invalid metric name: {metric_name}')


def compute_batch_average_metric(predictions: np.ndarray, ground_truth: np.ndarray, metric_name: str):
    batch_size = predictions.shape[0]

    y_true = ground_truth.flatten()
    y_pred = predictions.flatten()

    metric_results = []
    if len(predictions.shape) == 3:
        for i in range(batch_size):
            metric_results.append(compute_metric(y_true[i, :], y_pred[i, :], metric_name))
    else:
        metric_results.append(compute_metric(y_true, y_pred, metric_name))

    return np.mean(metric_results)

seed: int = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.clear_autocast_cache()
torch.cuda.empty_cache()
cuda: bool = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
CLASSIFIER_WEIGHTS_PATH = '../weights/PointNet2SemSegMsg-epoch-100.pth'

batch_size: int = 16
num_classes: int = 6
num_points: int = 4096
latent_dim: int = 1024
in_channels: int = 3
overwrite: bool = False

classifier = PointNet2SemSegMsg(num_classes=num_classes)
cls_state = torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device)
classifier.load_state_dict(cls_state)
classifier = classifier.to(device)
classifier.eval()

dataset_dir: str = '../../data/processed/'
_, test_files = get_train_val_file_paths(dataset_dir)

test_dataset = PointCloudInstancesDataset()
test_dataset.set_file_paths(test_files)

print(f'Loaded {len(test_dataset)} for testing!')
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

all_batch_acc = []
all_batch_f1 = []
with torch.no_grad():
    for points, labels in tqdm(test_loader, total=len(test_loader)):
        points = points.float().to(device)
        labels = labels.long().to(device)

        # Forward pass
        logits = classifier(points.transpose(2, 1))
        assert logits.shape == (batch_size, num_points, num_classes), 'Output does not match the expected shape!'

        predictions = logits.argmax(dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()

        batch_acc = compute_batch_average_metric(predictions, labels, 'balanced_accuracy_score')
        all_batch_acc.append(batch_acc)
        batch_f1 = compute_batch_average_metric(predictions, labels, 'f1_score')
        all_batch_f1.append(batch_f1)

    avg_balanced_acc = float(torch.tensor(all_batch_acc).mean())
    avg_f1_score = float(torch.tensor(all_batch_f1).mean())
    tqdm.write(f'Average balanced accuracy: {avg_balanced_acc:.4f}')
    tqdm.write(f'Average F1: {avg_f1_score:.4f}')