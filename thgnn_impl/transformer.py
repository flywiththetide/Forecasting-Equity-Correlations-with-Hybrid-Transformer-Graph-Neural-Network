
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [MaxLen, 1, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, D]
        x = x + self.pe[:x.size(0), :]
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, num_features, d_model=128, n_heads=8, num_layers=4, dropout=0.2, seq_len=30):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, dim_feedforward=d_model*2, activation='gelu', norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.d_model = d_model
        
        # Flatten [Seq, Batch, D] -> [Batch, Seq*D] and proj to Node Embedding
        self.output_proj = nn.Sequential(
            nn.Linear(seq_len * d_model, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU()
        )

    def forward(self, src):
        # src: [Batch=Nodes, SeqLen, Features]
        # Transformer expects [SeqLen, Batch, Features]
        src = src.permute(1, 0, 2) 
        
        src = self.input_proj(src) # [Seq, Nodes, D]
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src) # [Seq, Nodes, D]
        
        # Permute back to [Nodes, Seq, D]
        output = output.permute(1, 0, 2)
        
        # Flatten
        B, S, D = output.shape
        output = output.reshape(B, S * D)
        
        embedding = self.output_proj(output) # [Nodes, 512]
        return embedding
