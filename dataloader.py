import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch

class LatentsDataset(Dataset):
    def __init__(self, data_path, prompts, transform=None, target_transform=None):
        
        self.data_path = data_path
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):

        img_tensor_path = os.path.join(self.data_path, f"latent_{idx}.pt")
        print(img_tensor_path)
        img_tensor = torch.load(img_tensor_path)
        return img_tensor
    
# class LatentsDataloader(DataLoader):
#     def __init__(self, data_path, prompts, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
#         dataset = LatentsDataset(data_path, prompts)
#         super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

