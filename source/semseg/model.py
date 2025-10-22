"""
Code taken from GitHub-repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


def square_distance(source: torch.Tensor, destination: torch.Tensor):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    batch_size, n, _ = source.shape
    _, m, _ = destination.shape
    distance = -2 * torch.matmul(source, destination.permute(0, 2, 1))
    distance += torch.sum(source ** 2, -1).view(batch_size, n, 1)
    distance += torch.sum(destination ** 2, -1).view(batch_size, 1, m)

    return distance


def index_points(points: torch.Tensor, indices: torch.Tensor):
    """
    Input:
    points: input points training_data, [B, N, C]
    indices: sample index training_data, [B, S]
    Return:
        new_points: indexed points, [B, S, C]
    """
    device = points.device
    batch_size = points.shape[0]

    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indices.shape)

    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, indices, :]

    # print("points shape:", points.shape)
    # print("indices shape:", indices.shape)
    # print("indices max:", indices.max().item())

    return new_points


def farthest_point_sample(points: torch.Tensor, num_points: int):
    device = points.device
    batch_size, n, channels = points.shape

    centroids = torch.zeros(batch_size, num_points, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, n).to(device) * 1e10
    farthest = torch.randint(0, n, (batch_size,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)

    for i in range(num_points):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(batch_size, 1, 3)

        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, num_sample, xyz, new_xyz):
    device = xyz.device
    batch_size, src_dim, _ = xyz.shape
    _, new_dim, _ = new_xyz.shape
    group_indices = torch.arange(src_dim, dtype=torch.long).to(device).view(1, 1, src_dim).repeat(
        [batch_size, new_dim, 1])
    squared_distances = square_distance(new_xyz, xyz)

    group_indices[squared_distances > radius ** 2] = src_dim - 1
    group_indices = group_indices.sort(dim=-1)[0][:, :, :num_sample]

    group_first = group_indices[:, :, 0].view(batch_size, new_dim, 1).repeat([1, 1, num_sample])

    mask = group_indices == src_dim - 1
    group_indices[mask] = group_first[mask]

    # print("group_indices shape:", group_indices.shape)
    # print("group_indices max:", group_indices.max().item())
    # print("src_dim:", src_dim)

    return group_indices


def sample_and_group_all(point_coordinates: torch.Tensor, features: torch.Tensor):
    device = point_coordinates.device
    batch_size, num_points, in_channels = point_coordinates.shape
    new_point_coordinates = torch.zeros(batch_size, 1, in_channels).to(device)
    grouped_point_coordinates = point_coordinates.view(batch_size, 1, num_points, in_channels)

    if features is not None:
        new_features = torch.cat(
            [grouped_point_coordinates, features.view(batch_size, 1, num_points, -1)], dim=-1)
    else:
        new_features = grouped_point_coordinates

    return new_point_coordinates, new_features


def sample_and_group(num_points: int, radius, num_samples, point_coordinates,
                     features, return_feature_propagations: bool = False):
    batch_size, _, in_channels = point_coordinates.shape

    # [batch_size, num_points, in_channels]
    fps_idx = farthest_point_sample(point_coordinates, num_points)
    new_point_coordinates = index_points(point_coordinates, fps_idx)

    idx = query_ball_point(radius, num_samples, point_coordinates, new_point_coordinates)
    # [batch_size, num_points, num_samples, in_channels]
    grouped_point_coordinates = index_points(point_coordinates, idx)

    grouped_point_coordinates_norm = grouped_point_coordinates - new_point_coordinates.view(
        batch_size, num_points, 1, in_channels)

    if features is not None:
        grouped_features = index_points(features, idx)
        # [B, num_points, num_samples, C + D]
        new_features = torch.cat([grouped_point_coordinates_norm, grouped_features], dim=-1)
    else:
        new_features = grouped_point_coordinates_norm
    if return_feature_propagations:
        return new_point_coordinates, new_features, grouped_point_coordinates, fps_idx
    else:
        return new_point_coordinates, new_features

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, num_points: int, radius_list: "list[float]", num_sample_list: "list[int]",
                 in_channel: int, mlp_list: "list[list[int]]"):
        super(PointNetSetAbstractionMsg, self).__init__()

        self.num_points = num_points
        self.radius_list = radius_list
        self.num_sample_list = num_sample_list
        self.in_channel = in_channel
        self.mlp_list = mlp_list

        self.convolution_blocks = nn.ModuleList()
        self.batch_norm_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convolutions = nn.ModuleList()
            batch_norms = nn.ModuleList()

            previous_channel = in_channel

            for output_channel in mlp_list[i]:
                convolutions.append(nn.Conv2d(previous_channel, output_channel, 1))
                batch_norms.append(nn.BatchNorm2d(output_channel))
                previous_channel = output_channel

            self.convolution_blocks.append(convolutions)
            self.batch_norm_blocks.append(batch_norms)

    def forward(self, point_coordinates: torch.Tensor, features: torch.Tensor):
        point_coordinates = point_coordinates.permute(0, 2, 1)

        if features is not None:
            features = features.permute(0, 2, 1)

        batch_size, num_samples, in_channels = point_coordinates.shape
        num_points = self.num_points

        new_point_coordinates = index_points(point_coordinates, farthest_point_sample(point_coordinates, num_points))
        new_features_list = []

        for i, radius in enumerate(self.radius_list):
            k = self.num_sample_list[i]
            group_idx = query_ball_point(radius, k, point_coordinates, new_point_coordinates)
            grouped_point_coordinates = index_points(point_coordinates, group_idx)
            grouped_point_coordinates -= new_point_coordinates.view(batch_size, num_points, 1, in_channels)

            if features is not None:
                grouped_features = index_points(features, group_idx)
                grouped_features = torch.cat([grouped_features, grouped_point_coordinates], dim=-1)
            else:
                grouped_features = grouped_point_coordinates

            grouped_features = grouped_features.permute(0, 3, 2, 1)

            for j in range(len(self.convolution_blocks[i])):
                convolution = self.convolution_blocks[i][j]
                batch_norm = self.batch_norm_blocks[i][j]

                grouped_features = F.relu(batch_norm(convolution(grouped_features)))

            new_features = torch.max(grouped_features, 2)[0]
            new_features_list.append(new_features)

        new_point_coordinates = new_point_coordinates.permute(0, 2, 1)
        new_features_concat = torch.cat(new_features_list, dim=1)
        return new_point_coordinates, new_features_concat


class PointNetSetAbstraction(nn.Module):
    def __init__(self, num_points: Optional[int],
                 radius: Optional[float],
                 num_samples: Optional[int],
                 in_channels: int,
                 mlp: "list[int]",
                 group_all: bool):
        super(PointNetSetAbstraction, self).__init__()
        self.num_points = num_points
        self.radius = radius
        self.num_samples = num_samples
        self.group_all = group_all

        self.mlp_convolutions = nn.ModuleList()
        self.mlp_batch_norms = nn.ModuleList()

        previous_channel = in_channels
        for out_channel in mlp:
            self.mlp_convolutions.append(nn.Conv2d(previous_channel, out_channel, 1))
            self.mlp_batch_norms.append(nn.BatchNorm2d(out_channel))
            previous_channel = out_channel

    def forward(self, point_coordinates: torch.Tensor, features: torch.Tensor):
        point_coordinates = point_coordinates.permute(0, 2, 1)

        if features is not None:
            features = features.permute(0, 2, 1)

        if self.group_all:
            new_point_coordinates, new_features = sample_and_group_all(point_coordinates, features)
        else:
            new_point_coordinates, new_features = sample_and_group(self.num_points, self.radius, self.num_samples,
                                                                   point_coordinates, features)

        new_features = new_features.permute(0, 3, 2, 1)
        for idx, convolution in enumerate(self.mlp_convolutions):
            bn = self.mlp_batch_norms[idx]
            new_features = F.relu(bn(convolution(new_features)), inplace=True)

        new_features = torch.max(new_features, 2)[0]
        new_point_coordinates = new_point_coordinates.permute(0, 2, 1)
        return new_point_coordinates, new_features


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, mlp: "list[int]"):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convolutions = nn.ModuleList()
        self.mlp_batch_norms = nn.ModuleList()
        self.mlp = mlp

        previous_channel = in_channels

        for output_channel in self.mlp:
            convolution = nn.Conv1d(previous_channel, output_channel, 1)
            batch_norm = nn.BatchNorm1d(output_channel)
            self.mlp_convolutions.append(convolution)
            self.mlp_batch_norms.append(batch_norm)
            previous_channel = output_channel

    def forward(self, point_coordinates1: torch.Tensor, point_coordinates2: torch.Tensor,
                features1: torch.Tensor, features2: torch.Tensor):
        point_coordinates1 = point_coordinates1.permute(0, 2, 1)
        point_coordinates2 = point_coordinates2.permute(0, 2, 1)

        features2 = features2.permute(0, 2, 1)
        batch_size, num_samples, in_channels = point_coordinates1.shape
        _, num_points, _ = point_coordinates2.shape

        if num_points == 1:
            interpolated_features = features2.repeat(1, num_samples, 1)
        else:
            distances = square_distance(point_coordinates1, point_coordinates2)
            distances, idx = distances.sort(dim=-1)
            distances, idx = distances[:, :, :3], idx[:, :, :3]

            distances_reciprocal = 1.0 / (distances + 1e-8)  # Avoid division with 0

            # Divide the weights to have values between 0 and 1
            weights = distances_reciprocal / torch.sum(distances_reciprocal, dim=2, keepdim=True)

            interpolated_features = torch.sum(index_points(features2, idx) * weights.view(
                batch_size, num_samples, 3, 1), dim=2)

        if features1 is not None:
            features1 = features1.permute(0, 2, 1)
            new_features = torch.cat([features1, interpolated_features], dim=-1)
        else:
            new_features = interpolated_features

        new_features = new_features.permute(0, 2, 1)

        for i, convolution in enumerate(self.mlp_convolutions):
            batch_norm = self.mlp_batch_norms[i]
            new_features = F.relu(batch_norm(convolution(new_features)))

        return new_features
    
# PointNet++ with Multi-Scale Grouping
class PointNet2SemSegMsg(nn.Module):
    def __init__(self, num_classes: int, num_features: int = 0, use_logits: bool = False):
        super(PointNet2SemSegMsg, self).__init__()
        self.use_logits: bool = use_logits

        self.set_abstraction_msg1 = PointNetSetAbstractionMsg(
            num_points=1024,
            radius_list=[0.05, 0.1],
            num_sample_list=[16, 32],
            in_channel=6 + num_features,  # !! NOTE: Adapt based on the task and amount of features you have
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
            mlp_list=[[256, 256, 512], [256, 384, 512]]
        )

        self.feature_propagation4 = PointNetFeaturePropagation(
            in_channels=512 + 512 + 256 + 256,
            mlp=[256, 256]
        )

        self.feature_propagation3 = PointNetFeaturePropagation(
            in_channels=128 + 128 + 256,
            mlp=[256, 256]
        )

        self.feature_propagation2 = PointNetFeaturePropagation(
            in_channels=32 + 64 + 256,
            mlp=[256, 128]
        )

        self.feature_propagation1 = PointNetFeaturePropagation(
            in_channels=128,
            mlp=[128, 128, 128]
        )

        self.convolution1 = nn.Conv1d(128, 128, 1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.convolution2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, features: torch.Tensor):
        l0_features = features
        l0_coordinates = features[:, :3, :]  # First 3 columns should be the XYZ coordinates

        # NOTE: l0_features has XYZ + features
        # l0_coordinates has only XYZ, if feature dimensionality is 3, then in_channel is 9
        # XYZ (3 channels) AND XYZ (3 channels) + features (3 channels) = 9
        # Additional 3 features, would result in in_channels = 12
        # print(f"Shape l0_features: {l0_features.size()}; l0_coordinates: {l0_coordinates.size()}")
        l1_coordinates, l1_features = self.set_abstraction_msg1(l0_coordinates, l0_features)
        l2_coordinates, l2_features = self.set_abstraction_msg2(l1_coordinates, l1_features)
        l3_coordinates, l3_features = self.set_abstraction_msg3(l2_coordinates, l2_features)
        l4_coordinates, l4_features = self.set_abstraction_msg4(l3_coordinates, l3_features)

        l3_features = self.feature_propagation4(l3_coordinates, l4_coordinates, l3_features, l4_features)
        l2_features = self.feature_propagation3(l2_coordinates, l3_coordinates, l2_features, l3_features)
        l1_features = self.feature_propagation2(l1_coordinates, l2_coordinates, l1_features, l2_features)
        l0_features = self.feature_propagation1(l0_coordinates, l1_coordinates, None, l1_features)

        x = self.dropout(F.relu(self.batch_norm1(self.convolution1(l0_features))))
        x = self.convolution2(x)
        x = F.log_softmax(x, dim=1) if self.use_logits else F.softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        # If you need the features, return x, l4_features as tuple
        return x