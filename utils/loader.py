from dotenv import load_dotenv
import os 
load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')
MODEL_NAME = os.getenv('MODEL_NAME')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class ChunkedDataset(Dataset):
    def __init__(self, data_dir, chunk_id):
        self.data_dir = data_dir
        self.chunk_id = chunk_id
        self.data = self.load_chunk() # (50_000, 64, 2)

    def load_chunk(self):
        chunk_path = os.path.join(self.data_dir, f'train.{self.chunk_id:02}.ds.pt')
        return torch.load(chunk_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_data_loaders(data_dir, batch_size):
    # First, load only chunk 01 to compute scaling parameters
    first_chunk = ChunkedDataset(data_dir, 1)
    # Reshape to 2D array (all_samples, features) for StandardScaler
    first_chunk_data = first_chunk.data.reshape(-1, 2)
    
    # Initialize and fit StandardScaler, which will be used to scale the data
    # for all chunks
    scaler = StandardScaler()
    scaler.fit(first_chunk_data)
    
    # Now load all chunks and apply scaling
    datasets = []
    for i in range(1, 2):
        dataset = ChunkedDataset(data_dir, i)
        # Apply scaling to each chunk
        scaled_data = dataset.data.reshape(-1, 2)
        scaled_data = scaler.transform(scaled_data)
        # Reshape back to original shape
        dataset.data = torch.FloatTensor(scaled_data.reshape(-1, 64, 2))
        datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return data_loader, scaler

def create_attention_mask(data, num_heads):
    # 1. Mask NaN values (padding)
    nan_mask = torch.isnan(data).any(dim=-1)  # Shape: (batch_size, seq_len)

    # 2. Create autoregressive mask
    seq_len = data.size(1)
    autoregressive_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=data.device))  # Shape: (seq_len, seq_len)

    # 3. Combine masks
    # Expand dimensions for broadcasting
    nan_mask = nan_mask.unsqueeze(1)  # Shape: (batch_size, 1, seq_len)
    autoregressive_mask = autoregressive_mask.unsqueeze(0)  # Shape: (1, seq_len, seq_len)

    # Combine masks
    mask = nan_mask | ~autoregressive_mask  # Shape: (batch_size, seq_len, seq_len)

    # 4. Adjust mask for multi-head attention
    # Expand the mask to (batch_size, num_heads, seq_len, seq_len)
    mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

    # Reshape the mask to (num_heads * seq_len, batch_size, batch_size)
    mask = mask.view(mask.size(0), -1, mask.size(3), mask.size(3))
    mask = mask.transpose(0, 1).contiguous()
    mask = mask.view(-1, mask.size(2), mask.size(3))

    return mask

# Example usage (assuming you have batch_size defined in your .env file)
# batch_size = int(os.getenv('BATCH_SIZE'))
# train_loader = create_data_loaders(DATA_DIR, batch_size)

# Now you can use train_loader in your training loop
# For example:
# for batch in train_loader:
#     # Process your batch here
#     pass
