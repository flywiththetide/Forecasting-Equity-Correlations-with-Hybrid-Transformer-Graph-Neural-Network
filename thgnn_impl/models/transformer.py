
import torch
import torch.nn as nn
from .layers import PositionalEncoding

class TemporalEncoder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, num_layers=4, dropout=0.2, seq_len=30):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, dim_feedforward=d_model*2, activation='gelu', norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(seq_len * d_model, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU()
        )

    def forward(self, src):
        # src: [Batch=Nodes, SeqLen, Features]
        src = src.permute(1, 0, 2) 
        src = self.input_proj(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        B, S, D = output.shape
        output = output.reshape(B, S * D)
        embedding = self.output_proj(output) 
        return embedding
