import torch
import torch.nn as nn

# This is a GPT-like context encoder
# It takes a sequence of coordinates, returns a sequence of contexts, then only the last element
# is returned as the context vector

class ContextEncoder(nn.Module):
    def __init__(self, coord_dim, d_model, nhead, num_layers, dim_feedforward, scaler, dropout=0.1):
        super(ContextEncoder, self).__init__()
        
        self.scaler = scaler
        self.input_projection = nn.Linear(coord_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.d_model = d_model

    def set_scaler(self, scaler):
        """
        Sets the scaler used for normalizing input coordinates.
        
        Args:
            scaler: A fitted StandardScaler instance used to normalize coordinates
        """
        self.scaler = scaler

    def forward(self, src, mask=None):
        # Reshape to 2D for scaling
        batch_size, seq_len, feat_dim = src.shape
        src_reshaped = src.reshape(-1, feat_dim)
        
        # Convert to numpy, scale, and back to tensor
        src_scaled = torch.FloatTensor(
            self.scaler.transform(src_reshaped.cpu().numpy())
        ).to(src.device)
        
        # Reshape back to 3D
        src_scaled = src_scaled.reshape(batch_size, seq_len, feat_dim) # shape: (batch_size, seq_len, feat_dim) = (batch_size, 63, 2)
        
        # Project coordinates to d_model dimensions instead of using embedding
        src_projected = self.input_projection(src_scaled) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)) # shape: (batch_size, seq_len, d_model)
        # Permute dimensions to (seq_len, batch_size, d_model) because the transformer expects (seq_len, batch_size, d_model)
        src_projected = src_projected.permute(1, 0, 2) # shape: (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src_projected, mask=mask)
        # Permute dimensions back to (batch_size, seq_len, d_model)
        output = output.permute(1, 0, 2) # shape: (batch_size, seq_len, d_model)
        return output # shape: (batch_size, seq_len, d_model)