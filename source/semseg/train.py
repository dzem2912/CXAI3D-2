import os
import shutil
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score,
                             mutual_info_score)

from model import PointNet2SemSegMsg
from dataset import PointCloudInstancesDataset, get_train_val_file_paths


def augment_point_cloud(batch_data: np.ndarray,
                        scale_low: float = 0.8, scale_high: float = 1.25,
                        shift_range: float = 0.1,
                        jitter_sigma: float = 0.01,
                        jitter_clip: float = 0.05) -> np.ndarray:
    B, N, C = batch_data.shape
    out = batch_data.copy()

    # scale
    scales = np.random.uniform(scale_low, scale_high, size=(B,1,1))
    out *= scales

    # shift
    shifts = np.random.uniform(-shift_range, shift_range, size=(B,1,3))
    out += shifts

    # jitter
    noise = np.clip(jitter_sigma * np.random.randn(B, N, C),
                    -jitter_clip, jitter_clip)
    out += noise

    return out

def clear_directory(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

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

cuda: bool = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

learning_rate: float = 1e-6
weight_decay: float = 1e-4

num_points: int = 4096
num_epochs: int = 100
batch_size: int = 16
num_classes: int = 6
num_features: int = 0

MODEL_TYPE: str = 'PN2SSMSG_v2_norm'
LOSS_PATH: str = f'loss/{MODEL_TYPE}/'
WEIGHTS_PATH: str = f'weights/{MODEL_TYPE}/'


clear_directory(WEIGHTS_PATH)
clear_directory(LOSS_PATH)

criterion = nn.CrossEntropyLoss()

def val_epoch(network: PointNet2SemSegMsg,
              val_loader: DataLoader,
              idx: int):
    network.eval()
    epoch_loss = 0.0
    all_batch_acc = []

    weights = torch.tensor([0.782, 0.735, 2.177, 0.735, 0.735, 5.520], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for points, labels in tqdm(val_loader, total=len(val_loader)):
            points = points.float().to(device)
            labels = labels.long().to(device)

            # Forward pass
            logits = network(points.transpose(2,1))
            assert logits.shape == (batch_size, num_points, num_classes), 'Output does not match the expected shape!'

            loss = F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1), weight=weights)
            
            epoch_loss += loss.item()
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()

            batch_acc = compute_batch_average_metric(labels, preds, 'balanced_accuracy_score')
            all_batch_acc.append(batch_acc)
        
        avg_loss = epoch_loss / len(val_loader)

        avg_balanced_acc = float(torch.tensor(all_batch_acc).mean())
        tqdm.write(f'Average val balanced accuracy: {avg_balanced_acc:.4f}')

    return avg_loss

def train_epoch(network: PointNet2SemSegMsg, optimizer: optim.Optimizer, train_loader: DataLoader, idx: int = 0):
    network.train()

    total_loss = 0.0
    all_batch_acc = []

    weights = torch.tensor([0.782, 0.735, 2.177, 0.735, 0.735, 5.520], dtype=torch.float32).to(device)
    
    for points, labels in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()

        points = points.data.numpy()
        points = augment_point_cloud(points)
        points = torch.from_numpy(points)

        points = points.float().to(device)
        labels = labels.long().to(device)

        logits = network(points.transpose(2, 1))

        assert logits.shape == (batch_size, num_points, num_classes), f'Output does not match the expected shape! GOT: {logits.shape}; EXPECTED: {batch_size, num_points, num_classes}'

        loss = F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1), weight=weights)

        total_loss += loss.item()

        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()

        batch_acc = compute_batch_average_metric(labels, preds, 'balanced_accuracy_score')
        all_batch_acc.append(batch_acc)

        loss.backward()
        optimizer.step()

    avg_balanced_acc = float(torch.tensor(all_batch_acc).mean())
    avg_loss = total_loss / len(train_loader)

    print(f"Train loss per point: {avg_loss:.6f} — Balanced Acc: {avg_balanced_acc:.4f}")
    return avg_loss

def train():
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    
    dataset_dir: str = '../../data/synthetic/processed/'
    train_files, val_files = get_train_val_file_paths(dataset_dir)

    train_dataset = PointCloudInstancesDataset()
    train_dataset.set_file_paths(train_files)

    val_dataset = PointCloudInstancesDataset()
    val_dataset.set_file_paths(val_files)

    print(f'Loaded {len(train_dataset)} samples for training and {len(val_dataset)} for validation!')
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    network = PointNet2SemSegMsg(num_classes=num_classes, num_features=num_features)
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_list = []
    val_loss_list = []

    for idx in range(num_epochs):
        train_loss = train_epoch(network, optimizer, train_loader, idx)
        train_loss_list.append(train_loss)

        val_loss = val_epoch(network, val_loader, idx)
        val_loss_list.append(val_loss)

        if len(val_loss_list) > 5 and val_loss_list[-1] > max(val_loss_list[-5:]):
            print("Validation loss stopped improving, consider early stopping!!!!")

        print(f'Epoch {idx + 1}/{num_epochs}: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

        plt.plot(train_loss_list, label='train loss')
        plt.plot(val_loss_list, label='val loss')

        # Server cannot render - save visualizations
        plt.savefig(os.path.join(LOSS_PATH, f'training_val_loss_epoch_{idx}.png'))
        plt.cla()
        plt.cla()
        plt.close()

        if idx % 10 == 0 or idx == (num_epochs - 1):
            torch.save(network.state_dict(), os.path.join(WEIGHTS_PATH, f'PointNet2SemSegMsg-epoch-{idx}.pth'))

    torch.save(network.state_dict(), os.path.join(WEIGHTS_PATH, f'PointNet2SemSegMsg-epoch-{num_epochs}.pth'))

if __name__ == "__main__":
    train()