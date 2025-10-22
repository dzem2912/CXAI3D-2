import os
import sys
import shutil
import random
import networkx as nx
import torch
import numpy as np
import open3d as o3d
import umap
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.tri as mtri
import traceback

from matplotlib.collections import LineCollection
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from source.generative.model import PointCloudAE
from source.semseg.model import PointNet2SemSegMsg
from source.generative.dataset import get_train_val_file_paths
from source.generative.dataset import PointCloudInstancesDataset

SEM_CLASSES = {
    'bus': 0,
    'car': 1,
    'motorcycle': 2,
    'airplane': 3,
    'boat': 4,
    'tower': 5
}

PALETTE = np.array([
    [200, 50, 50],  # 0 bus
    [35, 142, 35],  # 1 car
    [0, 0, 255],  # 2 motorcycle
    [255, 165, 0],  # 3 airplane
    [255, 0, 255],  # 4 boat
    [120, 120, 120],  # 5 train
], dtype=np.uint8)

"""
[200, 50, 50] dark red
[35, 142, 35] forest green
[0, 0, 255]  pure blue
[255, 165, 0] orange
[255, 0, 255]  magenta
[120, 120, 120]  medium gray
[255, 255, 0]  yellow
"""


def plot_tradeoff(x, y, z, color='b', marker='o', output_dir=''):
    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length.")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color, marker=marker, s=30)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Validity")
    ax.set_zlabel("Similarity")
    ax.set_title("Trade-Off-3D")

    plt.savefig(f'{output_dir}/trade-off-3D-plot.jpg')
    plt.close()


def compute_sparsity_nn(original: np.ndarray, counterfactual: np.ndarray, eps: float) -> float:
    """
    Fraction of points in original point cloud that deviate more than eps
    when matched to their nearest neighbor in counterfactual point cloud.
    """
    if original.shape != counterfactual.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("Inputs must both be (N, 3) arrays of the same shape.")
    if original.size == 0:
        return 0.0

    eps2 = eps * eps
    try:
        tree = cKDTree(counterfactual)
        dists, _ = tree.query(original, k=1, workers=-1)
        return float((dists * dists > eps2).mean())
    except Exception:
        diff = original[:, None, :] - counterfactual[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        min_sq = sq.min(axis=1)
        return float((min_sq > eps2).mean())


def compute_validity_nn(original_points: np.ndarray, counterfactual_points: np.ndarray, softmax_orig: np.ndarray,
                        softmax_cf: np.ndarray) -> float:
    if original_points.shape[0] != softmax_orig.shape[0]:
        raise ValueError("original_xyz and softmax_orig must align in N")
    if counterfactual_points.shape[0] != softmax_cf.shape[0]:
        raise ValueError("counterfactual_xyz and softmax_cf must align in M")

    c_star = np.argmax(softmax_orig, axis=1)

    tree = KDTree(counterfactual_points)
    nn_idx = tree.query(original_points, k=1, return_distance=False).ravel()

    p_orig = softmax_orig[np.arange(len(c_star)), c_star]
    p_cf = softmax_cf[nn_idx, c_star]

    return float(np.mean(np.abs(p_cf - p_orig)))


def compute_similarity_nn(original: np.ndarray, counterfactual: np.ndarray) -> float:
    """
    Chamfer distance but normalized to [0, 1]
    """
    if original.ndim != 2 or counterfactual.ndim != 2 or original.shape[1] != counterfactual.shape[1]:
        raise ValueError("Both point clouds must have the same dimensions (N,3) and (M,3).")
    if original.size == 0 or counterfactual.size == 0:
        return 0.0

    tree1 = KDTree(original)
    dists1, _ = tree1.query(counterfactual, k=1, return_distance=True)
    tree2 = KDTree(counterfactual)
    dists2, _ = tree2.query(original, k=1, return_distance=True)

    cd = np.mean(dists1 ** 2) + np.mean(dists2 ** 2)
    return cd / (1.0 + cd)


def main(seed: int, first_class: str, second_class: str, results_dir: str = ''):
    print(f'Running experiment with random seed: {seed} for classes: {first_class} and {second_class}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.cuda.empty_cache()
    if results_dir == '':
        results_dir = 'results/'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    OUTPUT_DIR: str = os.path.join(results_dir, f'{first_class}_{second_class}_{seed}/')
    SOURCE_CLASS: int = SEM_CLASSES[first_class]
    TARGET_CLASS: int = SEM_CLASSES[second_class]

    DATA_DIR: str = '../data/synthetic/processed/'

    batch_size: int = 32

    def clear_directory(dir_path: str):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    clear_directory(OUTPUT_DIR)

    train_paths, val_paths = get_train_val_file_paths(DATA_DIR, split=0.4)

    train_dataset = PointCloudInstancesDataset()
    train_dataset.set_file_paths(train_paths)

    test_dataset = PointCloudInstancesDataset()
    test_dataset.set_file_paths(val_paths)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    print(f'Found {len(train_dataset)} train samples!')
    print(f'Found {len(test_dataset)} test samples!')

    AUTOENCODER_WEIGHTS_PATH = 'weights/AE-PCD-epoch-800.pth'
    CLASSIFIER_WEIGHTS_PATH = 'weights/PointNet2SemSegMsg-epoch-100.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes: int = 6
    num_points: int = 4096
    latent_dim: int = 1024
    in_channels: int = 3
    ALPHA: float = 1.0
    BETA: float = 1.0
    N_OTHER: int = 10
    overwrite: bool = False

    latents_path = os.path.join(OUTPUT_DIR, 'train_latents.pth')
    logits_path = os.path.join(OUTPUT_DIR, 'train_logits.pth')

    autoencoder = PointCloudAE(in_channels=in_channels, num_points=num_points, latent_dim=latent_dim)
    ae_state = torch.load(AUTOENCODER_WEIGHTS_PATH, map_location=device)
    autoencoder.load_state_dict(ae_state)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    classifier = PointNet2SemSegMsg(num_classes=num_classes)
    cls_state = torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device)
    classifier.load_state_dict(cls_state)
    classifier = classifier.to(device)
    classifier.eval()

    if not os.path.exists(latents_path) and not os.path.exists(logits_path) or overwrite:
        train_latents = []
        train_logits = []

        with torch.no_grad():
            for points, _ in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
                points = points.float().contiguous().to(device)
                points = points.transpose(2, 1)

                _, z = autoencoder(points)

                for batch_idx in range(z.size(0)):
                    single_z = z[batch_idx, :]
                    train_latents.append(single_z.detach().cpu())

                softmax_probabilities = classifier(points)
                for batch_idx in range(softmax_probabilities.size(0)):
                    single_softmax_probability = softmax_probabilities[batch_idx]
                    train_logits.append(single_softmax_probability.detach().cpu())

        train_latents = torch.stack(train_latents, dim=0)
        train_logits = torch.stack(train_logits, dim=0)
        train_logits = torch.mean(train_logits, dim=1)

        print(f'Shape Train Latents: {train_latents.shape}')
        print(f'Shape Train logits: {train_logits.shape}')

        torch.save(train_latents, os.path.join(OUTPUT_DIR, 'train_latents.pth'))
        torch.save(train_logits, os.path.join(OUTPUT_DIR, 'train_logits.pth'))
    else:
        train_latents = torch.load(latents_path).cpu()
        train_logits = torch.load(logits_path).cpu()

    umap_model = umap.UMAP(n_components=2, n_neighbors=5, metric='cosine', random_state=seed)
    embedding = umap_model.fit_transform(train_latents)

    triangulation = Delaunay(embedding)

    ###############################################################################################
    bandwidth: float = 0.5
    kde = KernelDensity(bandwidth=bandwidth).fit(embedding)
    log_density = kde.score_samples(embedding)
    density = np.exp(log_density)
    scaler = MinMaxScaler()
    normalized_densities = scaler.fit_transform(density.reshape(-1, 1)).flatten()
    density = normalized_densities

    train_logits = train_logits.data.numpy()

    target_class = TARGET_CLASS
    mask = np.all(
        train_logits[:, target_class][:, None] > np.delete(train_logits, target_class, axis=1),
        axis=1
    )

    valid_target_indices = np.where(mask)[0]

    G = nx.Graph()
    for idx in range(len(embedding)):
        G.add_node(idx, logits=train_logits[idx],
                   logit_target=train_logits[idx][TARGET_CLASS],
                   ypred=np.argmax(train_logits[idx]))

    for simplex in triangulation.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = simplex[i], simplex[j]
                d = np.linalg.norm(embedding[p1] - embedding[p2])

                weight_density_edge = 1 - ((density[p1] + density[p2]) / 2.0)
                weight_logit_edge = 1 - ((train_logits[p1][TARGET_CLASS] + train_logits[p2][TARGET_CLASS]) / 2.0)

                G.add_edge(p1, p2, distance_edge=d, weight_density=weight_density_edge,
                           weight_logit=weight_logit_edge,
                           weight=ALPHA * weight_density_edge + BETA * weight_logit_edge)

    # Set threshold to the 99th percentile of distances
    percentile: int = 99
    DISTANCE_THRESHOLD = np.percentile([G.edges[e]['distance_edge'] for e in G.edges()], percentile)
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['distance_edge'] > DISTANCE_THRESHOLD]
    G.remove_edges_from(edges_to_remove)

    ################################################################
    ################### Computing the Z path, not UMAP #############
    ################################################################
    from sklearn.neighbors import NearestNeighbors

    Z = train_latents.numpy()  # (N, latent_dim)
    k_z = 15  # try 10–30; tune as you like
    metric_z = 'cosine'  # or 'euclidean' (try both)

    # kNN neighbors in z
    nn = NearestNeighbors(n_neighbors=k_z + 1, metric=metric_z).fit(Z)
    dists, nbrs = nn.kneighbors(Z)  # includes self at index 0

    # KDE density in z (optional but mirrors UMAP density idea)
    kde_z = KernelDensity(bandwidth=0.5, metric='euclidean').fit(Z)
    log_dens_z = kde_z.score_samples(Z)
    dens_z = np.exp(log_dens_z)
    dens_z = MinMaxScaler().fit_transform(dens_z.reshape(-1, 1)).ravel()

    Gz = nx.Graph()
    for idx in range(len(Z)):
        Gz.add_node(idx,
                    logits=train_logits[idx],
                    logit_target=train_logits[idx][TARGET_CLASS],
                    ypred=int(np.argmax(train_logits[idx]))
                    )

    # add edges (skip self)
    for i in range(len(Z)):
        for j_idx in range(1, nbrs.shape[1]):
            j = nbrs[i, j_idx]
            # latent-space distance (use same metric as NearestNeighbors for consistency)
            if metric_z == 'cosine':
                # cosine "distance"
                dij = 1.0 - (np.dot(Z[i], Z[j]) / (np.linalg.norm(Z[i]) * np.linalg.norm(Z[j]) + 1e-12))
            else:
                dij = np.linalg.norm(Z[i] - Z[j])

            # mirror your UMAP weighting: combine (low) density + (low) target logit
            wd = 1.0 - 0.5 * (dens_z[i] + dens_z[j])
            wl = 1.0 - 0.5 * (train_logits[i][TARGET_CLASS] + train_logits[j][TARGET_CLASS])

            Gz.add_edge(
                i, j,
                distance_edge=dij,
                weight_density=wd,
                weight_logit=wl,
                weight=ALPHA * wd + BETA * wl
            )

    # (optional) prune the longest edges by latent distance, same idea as UMAP’s percentile pruning
    percentile_z = 99
    thr_z = np.percentile([Gz.edges[e]['distance_edge'] for e in Gz.edges()], percentile_z)
    Gz.remove_edges_from([(u, v) for u, v, d in Gz.edges(data=True) if d['distance_edge'] > thr_z])
    ################################################################
    #################### ############# ############# ############# #
    ################################################################

    # Visualization of graph and KDE

    """
    pos = embedding  # (N,2) array
    edges = np.array(list(G.edges()))
    segs = np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1)  # (E,2,2)

    # Optional: use edge weights for alpha (lighter for heavier weight)
    edge_w = np.array([G[u][v]['weight'] for u, v in edges])
    edge_w_norm = (edge_w - edge_w.min()) / (edge_w.max() - edge_w.min() + 1e-12)
    edge_alpha = 0.15 + 0.65 * (1.0 - edge_w_norm)  # higher weight → lighter

    w = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    print("edge weight stats: min=", w.min(), " max=", w.max(), " any NaN? ", np.isnan(w).any())

    # If you want to be strict:
    if not np.all(np.isfinite(w)) or (w < 0).any():
        bad = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if
               (not np.isfinite(d['weight'])) or (d['weight'] < 0)]
        raise ValueError(f"Found invalid edge weights (showing up to 5): {bad[:5]}")
    # -----------------------------
    # Evaluate KDE on a grid (right panel heatmap)
    # -----------------------------
    pad = 0.5
    (xmin, ymin), (xmax, ymax) = pos.min(0) - pad, pos.max(0) + pad
    grid_res = 300
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_res),
        np.linspace(ymin, ymax, grid_res)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    logp = kde.score_samples(grid)
    p = np.exp(logp).reshape(xx.shape)
    p_norm = (p - p.min()) / (p.max() - p.min() + 1e-12)

    # -----------------------------
    # Figure
    # -----------------------------
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])  # left: graph
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)  # right: KDE-colored points
    # cax = fig.add_subplot(gs[0, 2])                 # colorbar axis

    # Common limits/aspect/background
    for a in (ax0, ax1):
        a.set_xlim(xmin, xmax)
        a.set_ylim(ymin, ymax)
        a.set_aspect('equal', adjustable='box')  # same data aspect
        a.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # (1) Left: pruned graph
    lc = LineCollection(segs, linewidths=1.0)
    lc.set_alpha(edge_alpha if edge_alpha.ndim == 1 else 0.5)
    ax0.add_collection(lc)
    ax0.scatter(pos[:, 0], pos[:, 1], s=25,
                c=(PALETTE[np.argmax(train_logits, axis=1)] / 255.0),
                edgecolors='black', linewidths=0.5)
    ax0.set_title(f"UMAP Graph Pruned at the {percentile}th Percentile of Edge Length")
    ax0.set_xlabel("UMAP-1");
    ax0.set_ylabel("UMAP-2")

    # (2) Right: points only, colored by normalized KDE
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list('blue_red', ['#2166ac', '#b2182b'])
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)  # normalized_densities in [0,1]

    sc = ax1.scatter(
        pos[:, 0], pos[:, 1], s=25,
        c=normalized_densities, cmap=cmap, norm=norm,
        edgecolors='black', linewidths=0.5, alpha=0.9
    )
    ax1.set_title(f"KDE (bandwidth={bandwidth}) over UMAP Space")
    ax1.set_xlabel("UMAP-1");
    ax1.set_ylabel("UMAP-2")

    # Colorbar in its own axis (doesn't shrink ax1)
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Normalized KDE")
    # cbar = fig.colorbar(sc, cax=cax)
    # cbar.set_label("Normalized KDE")

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{first_class}_{second_class}_umap_graphKDE_side_by_side.jpg"),
        dpi=600, facecolor='white', edgecolor='none'
    )
    plt.close()"""
    ########################################################################################################################
    def_valid_target_indices = []

    for node in valid_target_indices:
        # Get neighbors of the node
        neighbors = list(G.neighbors(node))
        # Check if n_other neighbor is a valid target index
        n_other = 1
        if len(neighbors) < n_other:
            continue
        # Count how many neighbors are in valid_target_indices
        count = sum(1 for neighbor in neighbors if neighbor in valid_target_indices)
        # If at least n_other neighbor is in valid_target_indices, keep the node
        if count >= N_OTHER:
            def_valid_target_indices.append(node)

    test_latents = []
    test_logits = []

    test_latents_path = os.path.join(OUTPUT_DIR, 'test_latents.pth')
    test_logits_path = os.path.join(OUTPUT_DIR, 'test_logits.pth')

    if not os.path.exists(test_logits_path) or overwrite:
        with torch.no_grad():
            for points, _ in tqdm(test_dataloader, total=len(test_dataloader), smoothing=0.9):
                points = points.float().contiguous().to(device)
                points = points.transpose(2, 1)

                softmax_probabilities = classifier(points)
                for batch_idx in range(softmax_probabilities.size(0)):
                    test_logits.append(softmax_probabilities[batch_idx].detach().cpu())

        test_logits = torch.stack(test_logits, dim=0)
        test_logits = torch.mean(test_logits, dim=1)
        torch.save(test_logits, test_logits_path)
    else:
        test_logits = torch.load(test_logits_path).cpu()

    test_predictions = test_logits.argmax(dim=1).data.numpy()

    indices_source_class = np.where(test_predictions == SOURCE_CLASS)[0]

    if len(indices_source_class) == 0:
        raise ValueError(f"No test instances found with predicted class {SOURCE_CLASS}")

    # Randomly select one test instance from the source class
    INDEX_TEST = np.random.choice(indices_source_class)
    x_test = test_dataset[INDEX_TEST]
    input_tensor = x_test[0].float().contiguous().unsqueeze(0).transpose(2, 1).to(device)

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(x_test[0].data.numpy())
    o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f'original.pcd'), original_pcd)

    y_pred_test = classifier(input_tensor).mean(dim=1).argmax(dim=1)
    y_pred_test = y_pred_test.detach().cpu().numpy()

    print(f'y_pred_test: {y_pred_test}')
    print(f'y_pred_test shape: {y_pred_test.shape}')
    _, x_test_latent = autoencoder(input_tensor)
    x_test_latent = x_test_latent.detach().cpu().numpy()
    x_test_embedding = umap_model.transform(x_test_latent)

    distances = np.linalg.norm(embedding - x_test_embedding, axis=1)
    closest_index = np.argmin(distances)

    # Shortest paths from source to nodes with target class
    all_paths = nx.single_source_dijkstra_path(G, source=closest_index, weight='weight')
    all_costs = nx.single_source_dijkstra_path_length(G, source=closest_index, weight='weight')

    # Filter valid class-dominant targets based on indices with the target class

    target_paths = {i: all_paths[i] for i in all_paths if all_paths[i][-1] in def_valid_target_indices}
    target_costs = {i: all_costs[i] for i in all_paths if all_paths[i][-1] in def_valid_target_indices}

    # Find the min cost path to the target class
    if target_costs:
        closest_target = min(target_costs, key=target_costs.get)
        closest_path = target_paths[closest_target]
        path = embedding[target_paths[closest_target]]
    else:
        path = None
    #################################################################
    #################################################################
    # Source selection in z (parallel to UMAP source)
    x_test_lat = x_test_latent  # already computed (1, latent_dim)
    x_test_lat = x_test_lat.squeeze(0)

    # find nearest training latent as source node in z
    if metric_z == 'cosine':
        sims = Z @ x_test_lat / (np.linalg.norm(Z, axis=1) * np.linalg.norm(x_test_lat) + 1e-12)
        src_z = int(np.argmax(sims))
    else:
        dsrc = np.linalg.norm(Z - x_test_lat[None, :], axis=1)
        src_z = int(np.argmin(dsrc))

    paths_z = nx.single_source_dijkstra_path(Gz, source=src_z, weight='weight')
    costs_z = nx.single_source_dijkstra_path_length(Gz, source=src_z, weight='weight')

    # keep only paths that end in class-dominant targets
    target_paths_z = {i: paths_z[i] for i in paths_z if paths_z[i][-1] in def_valid_target_indices}
    target_costs_z = {i: costs_z[i] for i in costs_z if i in target_paths_z}

    if target_costs_z:
        target_z = min(target_costs_z, key=target_costs_z.get)
        path_z_idx = target_paths_z[target_z]  # node indices in training set (latent)
    else:
        path_z_idx = None
    #################################################################
    #################################################################
    try:
        print(f'LENGTH OF PATH VALUES: {len(path)}')
        fig, ax0 = plt.subplots(figsize=(7, 6))

        pos = np.asarray(embedding)
        path_idx = np.asarray(closest_path)

        z_path_idx = np.asarray(path_z_idx)
        z_path_segs = np.stack([pos[z_path_idx[:-1]], pos[z_path_idx[1:]]], axis=1)
        z_path_lc = LineCollection(
            z_path_segs,
            colors='tab:orange',
            linewidths=4.0,
            zorder=10,
            capstyle='round',
            joinstyle='round'
        )
        ax0.add_collection(z_path_lc)
        ax0.scatter(
            pos[z_path_idx, 0], pos[z_path_idx, 1],
            s=30, facecolors='none', edgecolors='tab:orange',
            linewidths=1.2, zorder=11
        )

        path_segs = np.stack([pos[path_idx[:-1]], pos[path_idx[1:]]], axis=1)
        path_lc = LineCollection(
            path_segs,
            colors='tab:blue',  # or '#1f77b4'
            linewidths=4.0,
            zorder=10,  # above the thin graph edges
            capstyle='round',
            joinstyle='round'
        )
        ax0.add_collection(path_lc)

        # optionally emphasize the path nodes (hollow blue circles)
        ax0.scatter(pos[:, 0], pos[:, 1], s=25,
                    c=(PALETTE[np.argmax(train_logits, axis=1)] / 255.0),
                    edgecolors='black', linewidths=0.5)
        ax0.scatter(
            pos[path_idx, 0], pos[path_idx, 1],
            s=30, facecolors='none', edgecolors='black', linewidths=1.2, zorder=11
        )

        # mark start (test point) and target explicitly (bigger, filled)
        ax0.scatter(
            x_test_embedding[0, 0], x_test_embedding[0, 1],
            s=30, color='tab:blue', edgecolors='black', linewidths=0.8,
            zorder=12, label='Source'
        )
        ax0.scatter(
            pos[closest_target, 0], pos[closest_target, 1],
            s=30, color='tab:orange', edgecolors='black', linewidths=0.8,
            zorder=12, label='Target'
        )

        # optional legend with proxy line for the path
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        # define your color-class mapping
        legend_elements = [
            mpatches.Patch(color='green', label='car'),
            mpatches.Patch(color='red', label='bus'),
            mpatches.Patch(color='yellow', label='airplane'),
            mpatches.Patch(color='magenta', label='boat'),
            mpatches.Patch(color='gray', label='tower'),
            mpatches.Patch(color='blue', label='motorcycle'),
            mlines.Line2D([], [], color='tab:blue', linewidth=3, label='Counterfactual Path')
        ]

        # assume ax0 is your axis
        ax0.legend(handles=legend_elements,
                   loc='upper left',  # put legend in upper right
                   frameon=True)
        plt.savefig(f'{OUTPUT_DIR}/{first_class}_{second_class}_umap_cfe_path.jpg', dpi=600)
        plt.close()

        if path is not None:
            n_steps = 3
            interpolated_path = []

            path_latent_indices = closest_path
            path = train_latents[path_latent_indices]

            # Get the test point latent
            p1 = torch.tensor(x_test_latent.squeeze(0))
            p2 = path[0]  # First latent in path

            for t in np.linspace(0, 1, n_steps + 1)[:-1]:
                interpolated_point = (1 - t) * p1 + t * p2
                interpolated_path.append(interpolated_point)

            # Intermediate counterfactuals
            for i in range(len(path) - 2):
                p1 = path[i]
                p2 = path[i + 1]
                for t in np.linspace(0, 1, n_steps + 2)[1:-1]:
                    interpolated_point = (1 - t) * p1 + t * p2
                    interpolated_path.append(interpolated_point)

            # last latent in path
            p1 = path[-2]
            p2 = path[-1]
            for t in np.linspace(0, 1, n_steps + 1)[1:]:
                interpolated_point = (1 - t) * p1 + t * p2
                interpolated_path.append(interpolated_point)

            # Stack all interpolated latent vectors
            interpolated_path = torch.stack(interpolated_path).to(device)

            decoder = autoencoder.decoder

            source_instance: np.ndarray = np.asarray([])
            source_softmax: np.ndarray = np.asarray([])

            similarities = []
            validities = []
            sparsities = []
            eps_list = [0.01, 0.05, 0.1, 0.5]
            sparsity_dict = {}

            for idx in range(interpolated_path.shape[0]):
                z = interpolated_path[idx].unsqueeze(0)

                with torch.no_grad():
                    output = decoder(z)

                    output_np = output.transpose(2, 1).squeeze(0).detach().cpu().numpy()

                    logits = classifier(output)

                labels = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
                logits = logits.squeeze(0).detach().cpu().numpy()

                colors = PALETTE[labels].astype(np.float32) / 255.0  # (N,3) in [0,1]

                if idx == 0:
                    source_instance = output_np
                    source_softmax = logits

                pcd_output = o3d.geometry.PointCloud()
                pcd_output.points = o3d.utility.Vector3dVector(output_np)
                pcd_output.colors = o3d.utility.Vector3dVector(colors)

                o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, f'output_{idx}.pcd'), pcd_output)

                for eps in eps_list:
                    sparsity_val = compute_sparsity_nn(source_instance, output_np, eps=eps)
                    print(f'Sparsity: {sparsity_val} for counterfactual: {idx} and eps={eps}')
                    if str(eps) not in sparsity_dict.keys():
                        sparsity_dict[str(eps)] = [sparsity_val]
                    else:
                        sparsities = sparsity_dict[str(eps)]
                        sparsities.append(sparsity_val)
                        sparsity_dict[str(eps)] = sparsities

                similarity_val = compute_similarity_nn(source_instance, output_np)
                similarities.append(similarity_val)
                validity_val = compute_validity_nn(source_instance, output_np, source_softmax, logits)
                validities.append(validity_val)

                print(f'Similarity: {similarity_val} for counterfactual: {idx}')
                print(f'Validity: {validity_val} for counterfactual: {idx}')

            similarities = np.asarray(similarities)
            validities = np.asarray(validities)
            sparsities = np.asarray(sparsities)

            N = len(similarities)
            t = np.arange(1, N + 1)
            VAL_THR: float = 0.5

            for eps in eps_list:
                sparsities = sparsity_dict[str(eps)]
                plt.plot(t, similarities, marker="o", label="Similarity")
                plt.plot(t, validities, marker="s", label="Validity")
                plt.plot(t, sparsities, marker="^", label="Sparsity")

                plt.xlabel("Interpolation step")
                plt.ylabel("Metric Score")
                plt.legend(loc='upper left')
                plt.tight_layout()

                plt.savefig(f"{OUTPUT_DIR}/{first_class}_{second_class}_metrics_eps_{eps}.jpg", dpi=300)
                plt.close()

            # plot_tradeoff(sparsities, validities, similarities, output_dir=OUTPUT_DIR)

            return similarities, sparsities, validities
    except Exception as e:

        print(f'A path was not found unfortunately!')
        print(e)
        traceback.print_exc()
        return [], [], []


#############################################################
# For quanititative analysis - multiple runs ################
# ###########################################################
import numpy as np
import matplotlib.pyplot as plt


def _interp_to_grid(series_list, G=100):
    g = np.linspace(0, 1, G)
    Y = []
    for y in series_list:
        y = np.asarray(y, float)
        yi = np.full_like(g, y[0]) if len(y) == 1 else np.interp(g, np.linspace(0, 1, len(y)), y)
        Y.append(yi)
    return g, np.vstack(Y)


def _band(Y, mode="iqr"):
    if mode == "sem":
        c = np.nanmean(Y, 0)
        s = np.nanstd(Y, 0, ddof=1) / np.sqrt(max(Y.shape[0], 1))
        lo, hi = c - s, c + s
    elif mode == "std":
        c = np.nanmean(Y, 0)
        s = np.nanstd(Y, 0, ddof=1)
        lo, hi = c - s, c + s
    else:  # "iqr"
        c = np.nanmedian(Y, 0)
        lo = np.nanpercentile(Y, 25, 0);
        hi = np.nanpercentile(Y, 75, 0)
    return c, np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def plot_metrics_clean(runs, metrics=("similarity", "validity", "sparsity"),
                       band="iqr", title=None, savepath=None):
    fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(7, 7))
    if len(metrics) == 1: axes = [axes]
    g = np.linspace(0, 1, 100)

    for ax, m in zip(axes, metrics):
        series = [r[m] for r in runs if m in r and len(r[m]) > 0]
        if not series:
            ax.set_visible(False);
            continue
        g, Y = _interp_to_grid(series, G=200)
        c, lo, hi = _band(Y, mode=band)

        # no spaghetti → much clearer
        ax.plot(g, c, lw=2, label=m)
        ax.fill_between(g, lo, hi, alpha=0.18, label=f"{m} {band.upper()}")

        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.legend()

    axes[-1].set_xlabel("Interpolation progress")
    if title: fig.suptitle(title, y=0.98)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=300); plt.close(fig)


seeds: list[int] = [1, 8, 24, 42, 56, 67, 72, 91, 144]


def run_experiments():
    runs = []
    first_class = 'car'
    second_class = 'tower'
    results_dir: str = 'results_multiple_runs/'
    for seed in seeds:
        run = {}
        similarity, sparsity, validity = main(seed, first_class, second_class, results_dir)
        if len(similarity) == 0 or len(sparsity) == 0 or len(validity) == 0:
            continue

        run['similarity'] = similarity
        run['sparsity'] = sparsity
        run['validity'] = validity
        runs.append(run)

    plot_metrics_clean(
        runs,
        metrics=("similarity", "validity", "sparsity"),
        band="sem",
        title=f"10 runs {first_class} vs {second_class}",
        savepath=f"{results_dir}/{first_class}_{second_class}_metrics_small_multiples.jpg")


########################################################################################

def run_pair():
    seed: int = 1
    first_class = 'car'
    second_class = 'bus'

    main(seed, first_class, second_class)


if __name__ == "__main__":
    run_pair()
