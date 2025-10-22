import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from kaolin.metrics.pointcloud import chamfer_distance

def masked_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:

    B, N, C = y_pred.shape

    # Flatten all tensors
    y_pred_flat = y_pred.view(-1, C)         # (B*N, C)
    y_true_flat = y_true.view(-1)            # (B*N,)
    mask_flat = mask_gt.view(-1).bool()      # (B*N,)

    # Apply mask
    y_pred_masked = y_pred_flat[mask_flat]   # (num_valid, C)
    y_true_masked = y_true_flat[mask_flat]   # (num_valid,)

    if y_true_masked.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    loss = F.cross_entropy(y_pred_masked, y_true_masked, reduction='mean')
    return loss

def masked_chamfer_emd_distance(y_pred: torch.Tensor, y_true: torch.Tensor, mask_gt: torch.Tensor, l_emd: float = 0.1) -> torch.Tensor:
    batch_size = y_pred.size(0)
    total_loss = 0.0

    for batch_idx in range(batch_size):
        valid_indices = mask_gt[batch_idx, :].bool()
        if valid_indices.sum() == 0:
            total_loss += 0.0
            continue
        
        x_true = y_true[batch_idx, valid_indices, :].unsqueeze(0)
        x_hat = y_pred[batch_idx, :, :].unsqueeze(0)

        cd_loss = chamfer_distance(x_hat, x_true)
        
        total_loss += cd_loss

        #x_hat = y_pred[batch_idx, valid_indices, :].unsqueeze(0)
        
        #emd_loss = emd_fn(x_hat, x_true)

        #total_loss += cd_loss + l_emd * emd_loss
    
    total_loss /= batch_size

    return total_loss

def masked_chamfer_distance(y_pred: torch.Tensor, y_true: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
    batch_size = y_pred.size(0)
    total_loss = 0.0

    for batch_idx in range(batch_size):
        valid_indices = mask_gt[batch_idx, :].bool()
        if valid_indices.sum() == 0:
            total_loss += 0.0
            continue

        #x_hat = y_pred[batch_idx, valid_indices, :].unsqueeze(0)
        x_hat = y_pred[batch_idx, :, :].unsqueeze(0)
        
        x_true = y_true[batch_idx, valid_indices, :].unsqueeze(0)

        total_loss += chamfer_distance(x_hat, x_true)
        #total_loss += emd_loss(x_hat, x_true)
    
    total_loss /= batch_size

    return total_loss


def random_scale_point_cloud(batch_data, scale_low: float = 0.8, scale_high: float = 1.25):
    batch_size, _, _ = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, batch_size)
    for batch_index in range(batch_size):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range: float = 0.1):
    batch_size, _, _ = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (batch_size, 3))
    for batch_index in range(batch_size):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data

def plot_latent_pca(z_list: list[torch.Tensor], epoch: int, dir_path: str, train: str):
    all_latents = torch.cat(z_list, dim=0)
    all_latents_np = all_latents.cpu().detach().numpy()
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(all_latents_np)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', alpha=0.7)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA of Latent Space")

    plt.savefig(os.path.join(dir_path, f'latent-pca-epoch_{epoch}_{train}.png'))
    plt.clf()
    plt.cla()
    plt.close()


def plot_latent_umap(z_list: list[torch.Tensor], epoch: int, dir_path: str, train: str):
    all_latents = torch.cat(z_list, dim=0)
    all_latents_np = all_latents.cpu().detach().numpy()

    reducer = UMAP(n_components=2, metric='cosine')

    latent_2d = reducer.fit_transform(all_latents_np)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', alpha=0.7, s=5)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"UMAP of Latent Space (epoch {epoch})")

    plt.savefig(os.path.join(dir_path, f'latent-umap-epoch_{epoch}_{train}.png'))
    plt.close()


def plot_latent_tsne(z_list: list[torch.Tensor], epoch: int, dir_path: str):
    all_latents = torch.cat(z_list, dim=0)
    all_latents_np = all_latents.cpu().detach().numpy()
    tsne = TSNE(n_components=2, perplexity=5, max_iter=1000)
    latent_2d_tsne = tsne.fit_transform(all_latents_np)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c='green', alpha=0.7)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE of Latent Space")

    plt.savefig(os.path.join(dir_path, f'latent-tsne-epoch_{epoch}.png'))
    plt.clf()
    plt.cla()
    plt.close()


def augment_point_cloud(batch_data: np.ndarray,
                        scale_low: float = 0.8, scale_high: float = 1.25,
                        shift_range: float = 0.1,
                        jitter_sigma: float = 0.01,
                        jitter_clip: float = 0.05) -> np.ndarray:
    """
    batch_data: (B, N, 3)
    """
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

if __name__ == '__main__':
    B, P, V = 3, 64, 32
    gt_batch = torch.randn(B, P, 3, device='cuda')
    pred_batch = torch.randn(B, P, 3, device='cuda')
    mask = torch.ones(B, P, device='cuda').bool()

    mask_zero = torch.zeros_like(mask).to('cuda')
    assert masked_chamfer_distance(pred_batch, gt_batch, mask_zero) == 0.0

    cd_true = chamfer_distance(pred_batch, gt_batch).sum() / B
    cd_masked = masked_chamfer_distance(pred_batch, gt_batch, mask)

    assert cd_true == cd_masked, f'Assertion failed: {cd_true} != {cd_masked}'