from dotenv import load_dotenv
import os 
load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')
MODEL_NAME = os.getenv('MODEL_NAME')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
batch_size = int(os.getenv('BATCH_SIZE'))

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from utils.loader import create_data_loaders, create_attention_mask

# DataLoader for training
train_loader, scaler = create_data_loaders(DATA_DIR, batch_size)
# Input: (batch_size, 64, 2)

import torch
from model.context_encoder import ContextEncoder
from model.cfm_model import CFMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize ContextEncoder with scalers
context_encoder = ContextEncoder(
    coord_dim=2,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    scaler=scaler,
    dropout=0.1
).to(device)

# Initialize CFMModel
cfm_model = CFMModel(
    input_dim=2,  # Assuming 2D coordinates (lat, lon)
    context_dim=128,  # Matching the output dimension of ContextEncoder
    hidden_dim=256,  # You can adjust this as needed
    num_blocks=4  # You can adjust this as needed
).to(device)

def train_model(optimizer, loss_fn):
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)

            # Generate attention mask
            mask = create_attention_mask(data)

            # Autoregressive training:
            # - Input:  [(lat1, lon1), (lat2, lon2), ...]
            # - Target: [(lat2, lon2), (lat3, lon3), ...] 
            #   (shifted by one position)
            
            # Create input and target sequences
            # input_seq shape: (batch_size, seq_len - 1, 2)
            # target_seq shape: (batch_size, seq_len - 1, 2)
            input_seq = data[:, :-1, :]  # Input: all but the last coordinate
            target_seq = data[:, 1:, :]  # Target: all but the first coordinate

            # Adjust mask for input sequence length
            # mask shape: (batch_size, seq_len - 1, seq_len - 1)
            mask = mask[:, :, :-1, :-1]

            # Forward pass with mask to obtain context embedding
            encoded_context = context_encoder(input_seq, mask=mask)
            # Context size: (batch_size, d_model), e.g., (batch_size, 128)

            # Generate a noise embedding of size input 
            x0 = torch.randn(batch_size, input_seq.shape[-1]).to(device)

            # Uniformly sample a time step between 0 and 1
            t = torch.rand((x0.shape[0], 1), device=x0.device)

            # Compute the loss
            optimizer.zero_grad()
            loss = loss_fn(cfm_model, x0, target_seq, t, encoded_context)
            loss.backward()
            optimizer.step()

            # Pass the noise with the context embedding through the generative flow model
            # Assuming you want to condition on the encoded context
            # Concatenate noise with context embedding along the last dimension
            # x0_with_context shape: (batch_size, seq_len - 1, d_model + 2)
            x0_with_context = torch.cat([x0.unsqueeze(1).repeat(1, input_seq.shape[1], 1), encoded_context], dim=-1)

            # Use generative flow model to sample the next coordinate
            # Input: (batch_size, seq_len, d_model)
            # Output: (batch_size, seq_len, 2)
            # You might need to adjust the sampling function based on your specific needs
            sampled_coords = cfm_model(x0_with_context, encoded_context)


