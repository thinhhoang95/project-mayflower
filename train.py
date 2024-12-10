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

import torch
from model.context_encoder import ContextEncoder
from model.conditional_flow import CFMModel, loss_fn 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize ContextEncoder with scalers
context_encoder = ContextEncoder(
    coord_dim=2,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    scaler=None,
    dropout=0.1
).to(device)

# Initialize CFMModel
cfm_model = CFMModel(
    input_dim=2,  # Assuming 2D coordinates (lat, lon)
    context_dim=128,  # Matching the output dimension of ContextEncoder
    hidden_dim=256,  
    num_blocks=4 
).to(device)

# Initialize optimizer
# Initialize optimizer with standard transformer learning rate and betas
optimizer = torch.optim.AdamW([
    {'params': context_encoder.parameters()},
    {'params': cfm_model.parameters()}
], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

# Learning rate scheduler with warmup
from torch.optim.lr_scheduler import LambdaLR

def get_lr_schedule(optimizer, warmup_steps=4000):
    def lr_lambda(step):
        # Linear warmup followed by inverse square root decay
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return (warmup_steps ** 0.5) * (step ** -0.5)
    
    return LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_schedule(optimizer)


# Print model summaries ==============================================
print("\nContext Encoder Architecture:")
print(context_encoder)
print("\nTotal parameters in Context Encoder:", sum(p.numel() for p in context_encoder.parameters()))

print("\nCFM Model Architecture:") 
print(cfm_model)
print("\nTotal parameters in CFM Model:", sum(p.numel() for p in cfm_model.parameters()))

print("\nTotal parameters in both models:", sum(p.numel() for p in context_encoder.parameters()) + sum(p.numel() for p in cfm_model.parameters()))
# =================================================================

# DataLoader for training
train_loader, scaler = create_data_loaders(DATA_DIR, batch_size)
# Input: (batch_size, 64, 2)

# Set scaler in context encoder
context_encoder.set_scaler(scaler)


def train_model(optimizer):
    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)

            # Generate attention mask
            mask = create_attention_mask(data, num_heads=8) # must match nhead in ContextEncoder

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
            mask = mask[:, :-1, :-1]

            # TRANSFORMER CONTEXT PASS ================================

            # Forward pass with mask to obtain context embedding
            encoded_context = context_encoder(input_seq, mask=mask) # shape: (batch_size, seq_len - 1, d_model)

            # Generate noise embedding with same shape as input sequence
            x0 = torch.randn_like(input_seq)  # shape: (batch_size, seq_len - 1, 2)

            # Uniformly sample a time step between 0 and 1 of shape (batch_size, seq_len - 1)
            t = torch.rand((input_seq.shape[0], input_seq.shape[1], 1), device=device)  # shape: (batch_size, seq_len - 1, 1)

            # CONDITIONAL FLOW MODEL PASS ================================
            x1 = target_seq
            t = torch.rand((x0.shape[0], x0.shape[1], 1), device=x0.device)
            optimizer.zero_grad()
            loss = loss_fn(cfm_model, x0, x1, t, encoded_context)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    print(f'Training model with {NUM_EPOCHS} epochs')
    train_model(optimizer)

