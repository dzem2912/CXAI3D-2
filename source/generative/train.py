import os
import torch
import shutil
import random
import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from kaolin.metrics.pointcloud import chamfer_distance

from torch.utils.data import DataLoader
from tqdm import tqdm
from model import PointCloudAE
from geomloss import SamplesLoss

from dataset import PointCloudInstancesDataset, get_train_val_file_paths
from train_utils import plot_latent_pca, plot_latent_umap, augment_point_cloud

seed: int = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


def clear_directory(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


cuda: bool = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

point_size: int = 4096
latent_dim: int = 1024
in_channels: int = 3

# Training Configuration
learning_rate: float = 1e-4
weight_decay: float = 1e-6
num_epochs: int = 1000
batch_size: int = 32

MODEL_TYPE: str = f'SYN-AE-{latent_dim}-PLUS-V2'
PREDS_PATH: str = f'preds/{MODEL_TYPE}/'
WEIGHTS_PATH: str = f'weights/{MODEL_TYPE}/'
LOSS_PATH: str = f'loss/{MODEL_TYPE}/'
LATENT_PCA_PATH: str = f'latent-pca/{MODEL_TYPE}/'
LATENT_UMAP_PATH: str = f'latent-umap/{MODEL_TYPE}/'

clear_directory(PREDS_PATH)
clear_directory(WEIGHTS_PATH)
clear_directory(LOSS_PATH)
clear_directory(LATENT_PCA_PATH)
clear_directory(LATENT_UMAP_PATH)

# Visualization configuration
num_samples: int = 3

assert num_samples < batch_size, "You cannot visualize more than you have"

emd_fn = SamplesLoss("sinkhorn", p=1, blur=1e-3, reach=1.0)

def save_n_samples(source: torch.Tensor, output: torch.Tensor, epoch_idx: int):
    for sample_index in range(num_samples):
        input_np = source.detach().cpu()[sample_index, :, :3].numpy()
        output_np = output.detach().cpu()[sample_index, :, :3].numpy()
        # Source point cloud
        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(input_np)

        o3d.io.write_point_cloud(os.path.join(PREDS_PATH, f'epoch_{epoch_idx}_input_{sample_index}.pcd'),
                                 pcd_input)

        # Predicted point cloud
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(output_np)
        o3d.io.write_point_cloud(os.path.join(PREDS_PATH, f'epoch_{epoch_idx}_prediction_{sample_index}.pcd'),
                                 pcd_pred)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        # If you use ReLU activations:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)


def val_epoch(network, val_loader: DataLoader, idx: int):
    network.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        total_samples: int = 0
        z_vectors = []

        for points, _ in tqdm(val_loader, total=len(val_loader), smoothing=0.9):
            total_samples += points.size(0)

            points = points.float().to(device)

            output, z = network(points.transpose(2, 1))
            output = output.transpose(2, 1)

            z_vectors.append(z)

            assert points.size() == (batch_size, point_size,
                                     3), f'Assertion failed: shape mismatch points.shape() != (B, N, 3)'
            assert output.size() == (batch_size, point_size,
                                     3), f'Assertion failed: shape mismatch output.shape() != (B, N, 3)'

            loss = chamfer_distance(output, points).sum()
            #loss = emd_fn(output.contiguous(), points.contiguous()).sum()

            epoch_loss += loss.item()

        if idx % 25 == 0:
            save_n_samples(points, output, idx)

        if idx % 25 == 0:
            plot_latent_pca(z_vectors, idx, LATENT_PCA_PATH, 'val')
            plot_latent_umap(z_vectors, idx, LATENT_UMAP_PATH, 'val')

            latent_tensor = torch.stack(z_vectors, dim=0)
            variance_per_dim = latent_tensor.var(dim=0)
            mean_variance = variance_per_dim.mean()

            print(f"Mean variance of latent space: {mean_variance.item():.6f}")

        return epoch_loss / total_samples


def train_epoch(network, optimizer: optim.Optimizer, train_loader: DataLoader, idx: int = 0):
    network.train()
    epoch_loss: float = 0.0
    total_samples: int = 0


    for points, _ in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        optimizer.zero_grad()
        total_samples += points.size(0)

        points = points.data.numpy()

        points = augment_point_cloud(points)

        points = torch.from_numpy(points).float().to(device)

        output, _ = network(points.transpose(2, 1))
        output = output.transpose(2, 1)

        assert points.size() == (batch_size, point_size, 3), f'Assertion failed: shape mismatch points.shape() != (B, N, 3): got {points.size()}'
        assert output.size() == (batch_size, point_size, 3), f'Assertion failed: shape mismatch output.shape() != (B, N, 3): got {output.size()}'

        loss = chamfer_distance(output, points).sum()
        #loss = emd_fn(output.contiguous(), points.contiguous()).sum()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / total_samples


def main():
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    root: str = '../../data/synthetic/processed/'

    train_files, val_files = get_train_val_file_paths(root, split=0.1)
    random.shuffle(train_files)
    

    train_dataset = PointCloudInstancesDataset(root=root)
    train_dataset.set_file_paths(train_files)

    val_dataset = PointCloudInstancesDataset(root=root)
    val_dataset.set_file_paths(val_files)

    print(f'Loaded {len(train_dataset)} samples for training and {len(val_dataset)} for validation!')
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    network = PointCloudAE(in_channels=in_channels, latent_dim=latent_dim, num_points=point_size)
    #network.apply(init_weights)
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

        print(f'Epoch {idx + 1}/{num_epochs}: train loss: {train_loss:.6f}, val loss: {val_loss:.6f}')

        plt.plot(train_loss_list, label='train loss')
        plt.plot(val_loss_list, label='val loss')

        # Server cannot render - save visualizations
        plt.savefig(os.path.join(LOSS_PATH, f'training_val_loss_epoch_{idx}.png'))
        plt.cla()
        plt.cla()
        plt.close()

        if idx % 10 == 0:
            torch.save(network.state_dict(), os.path.join(WEIGHTS_PATH, f'AE-PCD-epoch-{idx}.pth'))

    torch.save(network.state_dict(), os.path.join(WEIGHTS_PATH, f'AE-PCD-epoch-{num_epochs}.pth'))


if __name__ == "__main__":
    main()



