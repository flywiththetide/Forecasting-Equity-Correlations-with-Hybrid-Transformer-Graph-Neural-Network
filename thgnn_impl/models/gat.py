
import torch
import torch.nn as nn
from .layers import GATLayer

class RelationalEncoder(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATLayer(in_dim, hidden_dim, num_heads=heads))
            in_dim = hidden_dim 
            
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            out = layer(x, edge_index, edge_attr)
            x = self.norm(x + out)
        return x
