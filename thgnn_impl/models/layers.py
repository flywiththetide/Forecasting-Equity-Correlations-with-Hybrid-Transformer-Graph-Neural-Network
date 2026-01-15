
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

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, edge_dim=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W_node = nn.Linear(in_dim, out_dim)
        self.W_edge_attr = nn.Linear(edge_dim, out_dim)
        
        # Attention mechanism
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, D]
        N = x.size(0)
        
        # Project nodes
        h = self.W_node(x).view(N, self.num_heads, self.head_dim)
        
        # Project edges (resize to match heads)
        E = edge_index.size(1)
        e = self.W_edge_attr(edge_attr).view(E, self.num_heads, self.head_dim)
        
        idx_src, idx_dst = edge_index
        h_src = h[idx_src]
        h_dst = h[idx_dst]
        
        # Score
        scores = (h_src * self.att_src).sum(dim=-1) + (h_dst * self.att_dst).sum(dim=-1) + (e * self.att_edge).sum(dim=-1)
        scores = self.leaky_relu(scores)
        
        # Softmax approximation
        scores = scores - scores.max()
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(N, self.num_heads, device=x.device)
        sum_exp.index_add_(0, idx_dst, exp_scores)
        att_weights = exp_scores / (sum_exp[idx_dst] + 1e-6)
        
        # Aggregate
        out_msg = h_src * att_weights.unsqueeze(-1)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, idx_dst, out_msg)
        out = out.reshape(N, -1)
        return out
