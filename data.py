import torch
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    def __init__(self, num_samples, num_nodes, seed=None):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.points = torch.rand(num_samples, num_nodes, 2)
        self.distances = torch.cdist(self.points, self.points, p=2)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return {
            'points': self.points[idx],
            'distance': self.distances[idx]
        }

if __name__ == "__main__":
    dataset = TSPDataset(num_samples=5, num_nodes=20)
    sample = dataset[0]
    print("座標の形:", sample['points'].shape)
    print("距離行列の形:", sample['distance'].shape)
    print("データ生成成功！")