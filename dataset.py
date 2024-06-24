import torch
from torch.utils.data import Dataset


class GPDataset(Dataset):
    def __init__(self, data, num_tokens):
        self.data = data
        self.num_tokens = num_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, target = self.data[idx]
        return torch.tensor(window, dtype=torch.long), torch.tensor(target, dtype=torch.long)
