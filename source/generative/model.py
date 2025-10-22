"""
Parts of code on PointNet++ taken from GitHub-repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import PointTransformerConv
from torch_geometric.nn import knn_graph

from source.generative.pointnet_utils import PointNetSetAbstractionMsg

class PointNet2Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(PointNet2Encoder, self).__init__()
        self.latent_dim: int = latent_dim
        self.in_channels: int = in_channels
        
        self.set_abstraction_msg1 = PointNetSetAbstractionMsg(
            num_points=1024,
            radius_list=[0.05, 0.1],
            num_sample_list=[16, 32],
            in_channel=in_channels,
            mlp_list=[[16, 16, 32], [32, 32, 64]]
        )

        self.set_abstraction_msg2 = PointNetSetAbstractionMsg(
            num_points=256,
            radius_list=[0.1, 0.2],
            num_sample_list=[16, 32],
            in_channel=32 + 64 + 3,
            mlp_list=[[64, 64, 128], [64, 96, 128]]
        )

        self.set_abstraction_msg3 = PointNetSetAbstractionMsg(
            num_points=64,
            radius_list=[0.2, 0.4],
            num_sample_list=[16, 32],
            in_channel=128 + 128 + 3,
            mlp_list=[[128, 196, 256], [128, 196, 256]]
        )

        self.set_abstraction_msg4 = PointNetSetAbstractionMsg(
            num_points=16,
            radius_list=[0.4, 0.8],
            num_sample_list=[16, 32],
            in_channel=256 + 256 + 3,
            mlp_list=[[256, 256, self.latent_dim // 2], [256, 384, self.latent_dim // 2]]
        )

    def forward(self, features: torch.Tensor):
        l0_features = features
        l0_coordinates = features[:, :3, :]

        l1_coords, l1_feats = self.set_abstraction_msg1(l0_coordinates, l0_features)
        l2_coords, l2_feats = self.set_abstraction_msg2(l1_coords, l1_feats)
        l3_coords, l3_feats = self.set_abstraction_msg3(l2_coords, l2_feats)
        _, l4_feats = self.set_abstraction_msg4(l3_coords, l3_feats)

        latent = torch.max(l4_feats, dim=2, keepdim=True)[0]
        latent = latent.view(-1, self.latent_dim)

        return latent
    
class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(PointNetEncoder, self).__init__()
        self.in_channels: int = in_channels
        self.latent_dim: int = latent_dim

        self.conv1 = nn.Conv1d(in_channels, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, self.latent_dim, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.latent_dim)
        
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.bn4(self.conv4(x))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, self.latent_dim)

        return x

class FoldingNetDecFold1(nn.Module):
    def __init__(self, latent_dim: int, grid_dim: int = 2):
        super(FoldingNetDecFold1, self).__init__()
        self.latent_dim: int = latent_dim
        self.grid_dim: int = grid_dim

        self.conv1 = nn.Conv1d(self.latent_dim + self.grid_dim, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 3, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x

class FoldingNetDecFold2(nn.Module):
    def __init__(self, latent_dim: int, grid_dim: int = 2):
        super(FoldingNetDecFold2, self).__init__()
        self.latent_dim: int = latent_dim
        self.grid_dim: int = grid_dim

        self.conv1 = nn.Conv1d(self.latent_dim + self.grid_dim + 1, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 3, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))  
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x

def GridSamplingLayer(batch_size, meshgrid):
    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    num_points = np.prod([it[2] for it in meshgrid])
    grid = np.zeros((num_points, ndim), dtype=np.float32)
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    grid = np.repeat(grid[np.newaxis, ...], batch_size, axis=0)
    return grid


class FoldingNetDecoder(nn.Module):
    def __init__(self, latent_dim: int, grid_size: int):
        super(FoldingNetDecoder, self).__init__()
        self.latent_dim: int = latent_dim
        self.grid_size: int = grid_size
        self.fold1 = FoldingNetDecFold1(latent_dim=self.latent_dim)
        self.fold2 = FoldingNetDecFold2(latent_dim=self.latent_dim)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        num_grid = self.grid_size * self.grid_size

        x = x.unsqueeze(1).repeat(1, num_grid, 1)
        code = x.transpose(2, 1)

        meshgrid = [[-0.3, 0.3, self.grid_size], [-0.3, 0.3, self.grid_size]]
        grid = GridSamplingLayer(batch_size, meshgrid)
        grid = torch.from_numpy(grid).to(x.device)
        grid = grid.transpose(2, 1)

        x = torch.cat((code, grid), 1)

        x_fold1 = self.fold1(x)
        p1 = x_fold1

        x = torch.cat((code, x_fold1), 1)
        x_fold2 = self.fold2(x)

        return x_fold2

class PointTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dim: int = 128, k:int = 16):
        super().__init__()
        self.k = k

        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pt1 = PointTransformerConv(hidden_dim, hidden_dim)
        self.pt2 = PointTransformerConv(hidden_dim, hidden_dim)
        self.pt3 = PointTransformerConv(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous()

        latents = []

        for b in range(B):
            pos = x[b, :, :3]
            feats = x[b]

            edge_index = knn_graph(pos, k=self.k, batch=None)

            h = self.input_mlp(feats)
            h = self.pt1(h, pos, edge_index)
            h = self.pt2(h, pos, edge_index)
            h = self.pt3(h, pos, edge_index)

            global_feat = h.max(dim=0)[0]
            latents.append(global_feat)

        return torch.stack(latents, dim=0)
    
class SeedFormerDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_points: int, hidden_dim: int = 128, num_seed_points: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_seed_points = num_seed_points
        self.output_points = output_points
        self.hidden_dim = hidden_dim

        self.seed_proj = nn.Linear(latent_dim, num_seed_points * hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(output_points, hidden_dim))

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=False, dropout=0.3),
            num_layers=1
        )

        self.out_proj = nn.Linear(hidden_dim, 3)

    def forward(self, z: torch.Tensor):
        B = z.shape[0]
        seeds = self.seed_proj(z).view(B, self.num_seed_points, self.hidden_dim)

        memory = seeds.permute(1, 0, 2)

        tgt = self.pos_embed.unsqueeze(1).repeat(1, B, 1)

        # Decode
        decoded = self.transformer(tgt, memory)
        decoded = decoded.permute(1, 0, 2)

        # Project to 3D coordinates
        out = self.out_proj(decoded)
        out = out.transpose(2, 1)

        return out

class DecoderWithSpherePrior(nn.Module):
    def __init__(self, latent_dim: int, grid_res: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_res = grid_res
        self.num_points = grid_res ** 3

        #sphere_grid = self._generate_sphere_surface(self.num_points)
        #self.register_buffer('grid', sphere_grid)
        self.grid = nn.Parameter(self._generate_sphere_surface(self.num_points))

        self.input_dim = self.latent_dim + 3

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
            #nn.ReLU(inplace=True),
            #nn.Linear(128, 3)
        )

        self.refinement = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )

    def _generate_sphere_surface(self, num_points: int) -> torch.Tensor:
        phi = torch.rand(num_points) * 2 * torch.pi
        cos_theta = torch.rand(num_points) * 2 - 1
        theta = torch.acos(cos_theta)

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        return torch.stack([x, y, z], dim=-1)

    def forward(self, z: torch.Tensor):
        B = z.size(0)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)
        z_expanded = z.unsqueeze(1).expand(-1, self.num_points, -1)

        x = torch.cat([z_expanded, grid], dim=-1)

        out = self.mlp(x)
        out = out.transpose(2, 1)

        out = self.refinement(out)

        return out


class PointCloudAE(nn.Module):
    def __init__(self, in_channels: int, num_points: int, latent_dim: int):
        super(PointCloudAE, self).__init__()

        self.latent_dim: int = latent_dim
        self.num_points: int = num_points

        self.encoder = PointNet2Encoder(in_channels=in_channels + 3, latent_dim=self.latent_dim)
        self.decoder = DecoderWithSpherePrior(latent_dim=latent_dim)


    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction, z
