import os
import torch

from tqdm import tqdm

SEM_CLASSES = {
    'bus': 0,
    'car': 1,
    'motorcycle': 2,
    'airplane': 3,
    'boat': 4,
    'tower': 5
}

def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    input_dir: str = '../../data/synthetic/original/'
    output_dir: str = '../../data/synthetic/processed/'
    create_dir(output_dir)

    for sem_class in tqdm(SEM_CLASSES, total=len(SEM_CLASSES), smoothing=0.9):
        sem_input_dir: str = os.path.join(input_dir, sem_class)
        create_dir(os.path.join(output_dir, sem_class))

        sample_names: list[str] = os.listdir(sem_input_dir)

        for sample_name in tqdm(sample_names, total=len(sample_names), smoothing=0.9):
            sample_path: str = os.path.join(sem_input_dir, sample_name)

            data = torch.load(sample_path)

            points = data['points']

            points = points - points.mean(dim=0, keepdim=True)
            scale = torch.max(torch.norm(points, dim=1))
            points = points / scale
            
            sem_label = data['sem_label']

            label_id = SEM_CLASSES[sem_label]

            labels = torch.full((points.shape[0],), label_id, dtype=torch.long)

            data = {
                'points': points,
                'sem_labels': labels
            }

            output_path: str = os.path.join(os.path.join(output_dir, sem_class), sample_name)
            torch.save(data, output_path)
    

if __name__ == "__main__":
    main()