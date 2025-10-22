import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def get_train_val_file_paths(root: str, split: float):
    class_labels = os.listdir(root)
    print(f'CLASS LABELS: {class_labels}')
    val_file_paths = []
    train_file_paths = []

    for class_label in tqdm(class_labels, total=len(class_labels), smoothing=0.9):

        class_instances_dir: str = os.path.join(root, class_label)
        class_instances_paths: list[str] = [os.path.join(class_instances_dir, f) for f in os.listdir(class_instances_dir) if f.endswith('.pth')]

        counter = int(split * len(class_instances_paths))

        for class_instance_path in tqdm(class_instances_paths, total=len(class_instances_paths), smoothing=0.9):
            if counter != 0:
                val_file_paths.append(class_instance_path)
                counter -= 1
            else:
                train_file_paths.append(class_instance_path)
        print(f'{class_label}: TRAIN SAMPLES: {len(train_file_paths)}; VAL SAMPLES: {len(val_file_paths)}')
    
    return train_file_paths, val_file_paths

class PointCloudInstancesDataset(Dataset):
    def __init__(self, root: str = ''):
        super(PointCloudInstancesDataset, self).__init__()

        self.root: str = root
        self.file_paths = []

    def set_file_paths(self, file_paths: list[str]):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample_path: str = self.file_paths[idx]
        data = torch.load(sample_path)
        
        return data['points'], data['sem_labels']

if __name__ == '__main__':
    train_files, val_files = get_train_val_file_paths('../../data/synthetic/processed/')
    dataset = PointCloudInstancesDataset()
    dataset.set_file_paths(train_files)

    print(f'len: {len(dataset)}')
    print(dataset[0])