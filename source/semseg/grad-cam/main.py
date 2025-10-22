import os
import torch
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

from source.semseg.model import PointNet2SemSegMsg

SEM_CLASSES = {
    'bus': 0,
    'car': 1,
    'motorcycle': 2,
    'airplane': 3,
    'boat': 4,
    'tower': 5
}

torch.random.manual_seed(42)
np.random.seed(42)

class GradCAMPointNet2:
    def __init__(self, model: PointNet2SemSegMsg, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out
        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            print(f"Shape of gradient output: {type(self.gradients)}")

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def generate_targeted(
        self,
        input_tensor: torch.Tensor, # (B, 3, N)
        target_class: int,
        mask_non_target: bool,
        use_margin: bool,
        margin: float
    ):
        """
        Returns: numpy array (B, N) CAM for class K (targeted).
        """
        self.model.eval()
        with torch.enable_grad():
            output = self.model(input_tensor) # (B, N, C)
            if output.dim() != 3:
                raise ValueError("Model must return (B,N,C) logits.")

            B, N, C = output.shape
            device = output.device

            with torch.no_grad():
                pred = output.argmax(dim=-1)  # (B, N)

            cam_maps = []

            for b in range(B):
                """
                majority_cls = torch.bincount(pred[b], minlength=C).argmax().item()
                cls = majority_cls

                if use_margin:
                    mask_others = torch.ones(C, dtype=torch.bool, device=device)
                    mask_others[cls] = False
                    rival_max = output[b, :, mask_others].max(dim=-1).values
                    score = output[b, :, cls] - rival_max
                    if margin > 0.0:
                        score = torch.clamp(score - margin, min=0.0)
                else:
                    score = output[b, :, cls]

                if mask_non_target:
                    M = (pred[b] != cls)
                    if M.any():
                        Y = score[M].sum()
                    else:
                        Y = score.sum()
                else:
                    Y = score.sum()
                """
                if use_margin:
                    mask_others = torch.ones(C, dtype=torch.bool, device=device)
                    mask_others[target_class] = False
                    rival_max = output[b, :, mask_others].max(dim=-1).values
                    score = output[b, :, target_class] - rival_max
                    if margin > 0.0:
                        score = torch.clamp(score - margin, min=0.0)
                else:
                    score = output[b, :, target_class]

                if mask_non_target:
                    M = (pred[b] != target_class)
                    if M.any():
                        Y = score[M].sum()
                    else:
                        Y = score.sum()
                else:
                    Y = score.sum()

                self.model.zero_grad(set_to_none=True)
                Y.backward(retain_graph=True)

                if self.activations is None or self.gradients is None:
                    raise RuntimeError("Hooks didn't capture activations/gradients. Check target_layer.")

                A = self.activations[b]
                G = self.gradients[b]

                alpha = G.mean(dim=1, keepdim=True)

                #cam = torch.relu((alpha * A).sum(dim=0))
                cam = (alpha * A).sum(dim=0)
                #print(f'CAM MAX: {cam.max()}')

                if cam.min() < 0.0:
                    print(f'cam is negative!')

                print(cam.min())
                print(cam.max())

                cam = cam / cam.abs().max().clamp_min(1e-6)  # normalize to [-1, 1], preserving 0
                cam = (cam + 1) / 2  # map to [0, 1], with 0 → 0.5

                # Normalize for visualization
                #cam = cam - cam.min()
                #cam = cam / cam.max().clamp_min(1e-6)

                cam_maps.append(cam.detach().cpu().numpy())

        return np.stack(cam_maps, axis=0)

def save_point_cloud_with_cam(points: np.ndarray, cam: np.ndarray, filename: str):
    colors = ["darkblue", "gray", "crimson"]
    #colors = ["darkblue", "silver", "firebrick"]
    cmap = LinearSegmentedColormap.from_list("blue_gray_red", colors)
    colors = cmap(cam)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd)

def main():
    CLASS_INSTANCES_DIR: str = 'car_bus/'
    TARGET_CLASS: str = 'bus'

    CLASS_INS_OUTPUT_DIR: str = 'car_bus_out/'
    if not os.path.exists(CLASS_INSTANCES_DIR):
        os.makedirs(CLASS_INSTANCES_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSIFIER_WEIGHTS_PATH = '../../weights/PointNet2SemSegMsg-epoch-100.pth'

    model = PointNet2SemSegMsg(num_classes=6).to(device)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device))
    model.eval()

    class_instances_paths: list[str] = [os.path.join(CLASS_INSTANCES_DIR, f) for f in os.listdir(CLASS_INSTANCES_DIR) if f.endswith('.pcd')]
    class_instances_paths.sort()

    for class_instance_path in class_instances_paths:
        name: str = class_instance_path.split('/')[-1].replace('.pcd', '')
        points = np.asarray(o3d.io.read_point_cloud(class_instance_path).points)
        points = torch.tensor(points, dtype=torch.float32).to(device)
        with torch.no_grad():
            points = points.unsqueeze(0).to(device)
            grad_cam = GradCAMPointNet2(model=model, target_layer=model.feature_propagation1)
            cams = grad_cam.generate_targeted(points.transpose(2, 1), target_class=SEM_CLASSES[TARGET_CLASS],
                                              mask_non_target=True, use_margin=True, margin=0.0)
            points = points.squeeze(0).detach().cpu().numpy()

        cams = cams.reshape(-1)
        print(cams.min(), cams.max())
        grad_cam.remove_hooks()

        # Save to PCD
        save_path: str = os.path.join(CLASS_INS_OUTPUT_DIR, f'{name}.pcd')
        save_point_cloud_with_cam(points, cams, save_path)

        print(f"Saved Grad-CAM visualization to {save_path}")

if __name__ == '__main__':
    main()