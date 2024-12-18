import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
load_dotenv()

# Model parameters from environment variables
dataset_path = os.getenv('DATASET_PATH')
output_dir = os.getenv('OUTPUT_DIR')
num_epochs = int(os.getenv('NUM_EPOCHS'))
batch_size = int(os.getenv('BATCH_SIZE'))
learning_rate = float(os.getenv('LEARNING_RATE'))
num_samples = int(os.getenv('NUM_SAMPLES'))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.layers(x)

# Conditional Flow Matching Model
class CFMModel(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, num_blocks=3):
        super(CFMModel, self).__init__()
        self.input_proj = nn.Linear(input_dim + context_dim + 1, hidden_dim)
        
        # Stack of ResBlocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Final layer norm and output projection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x, c):
        # t: (batch, seq_length, 1)
        # x: (batch, seq_length, input_dim)
        # c: (batch, seq_length, context_dim)
        
        # Concatenate t, x, and c
        txc = torch.cat([t, x, c], dim=-1)  # (batch, seq_length, input_dim + context_dim + 1)
        
        # Apply input projection
        h = self.input_proj(txc)  # (batch, seq_length, hidden_dim)
        
        # Apply ResBlocks
        for res_block in self.res_blocks:
            h = res_block(h)  # (batch, seq_length, hidden_dim)
            
        # Final normalization and projection
        h = self.layer_norm(h)  # (batch, seq_length, hidden_dim)
        return self.output_proj(h)  # (batch, seq_length, input_dim)

# Optimal Transport Conditional Vector Field
def ot_conditional_vector_field(t, x, x0, x1):
    return (1 - t) * x0 + t * x1

# Loss Function
def loss_fn(model, x0, x1, t, c):
    x_t = (1 - t) * x0 + t * x1
    u_t = model(t, x_t, c)
    v_t = ot_conditional_vector_field(t, x_t, x0, x1)
    return torch.mean((u_t - v_t) ** 2)

# Training Function
def train_cfm(model, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        for batch in data_loader:
            x0, x1, c = batch
            # t: (batch, seq_length, 1) - assumed to be provided externally
            t = torch.rand((x0.shape[0], x0.shape[1], 1), device=x0.device)
            optimizer.zero_grad()
            loss = loss_fn(model, x0, x1, t, c)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# Sampling Function
def sample_from_model(model, num_samples, x_shape, device, steps=100):
    # x_shape: (seq_length, input_dim)
    x = torch.randn(num_samples, *x_shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        # t: (batch, seq_length, 1) - assumed to be provided externally
        t = torch.tensor(1.0 - i * dt, device=device).view(1, 1, 1).repeat(num_samples, x_shape[0], 1)
        with torch.no_grad():
            x = x + model(t, x) * dt
    return x

# # Data Generation
# def generate_data(num_samples, output_path):
#     # Example: mixture of two Gaussians
#     mean1 = torch.tensor([1.0, 2.0])
#     cov1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
#     mean2 = torch.tensor([-1.0, -2.0])
#     cov2 = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])

#     data1 = torch.distributions.MultivariateNormal(mean1, cov1).sample((num_samples // 2,))
#     data2 = torch.distributions.MultivariateNormal(mean2, cov2).sample((num_samples // 2,))
#     data = torch.cat([data1, data2], dim=0)

#     # Save data as numpy array
#     np.save(output_path, data.numpy())

#     return data

# # Load or generate data
# if os.path.exists(dataset_path + '.npy'):
#     data = torch.tensor(np.load(dataset_path + '.npy'), dtype=torch.float32)
# else:
#     data = generate_data(num_samples, dataset_path)

# # Prepare data loader
# x0 = torch.randn(num_samples, 2)  # Noise data
# x1 = data  # Real data
# train_dataset = TensorDataset(x0, x1)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Initialize model, optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CFMModel(input_dim=2, hidden_dim=128, num_blocks=3).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# train_cfm(model, train_loader, optimizer, num_epochs)

# # Sample from the model
# samples = sample_from_model(model, num_samples, x1.shape[1:], device)

# # Plotting
# plt.figure(figsize=(8, 8))
# plt.scatter(x1[:, 0].cpu(), x1[:, 1].cpu(), alpha=0.5, label='Real Data')
# plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.5, label='Generated Samples')
# plt.legend()
# plt.title('CFM Generated Samples vs Real Data')
# plt.savefig(os.path.join(output_dir, 'samples_vs_real.png'))
# plt.show() 