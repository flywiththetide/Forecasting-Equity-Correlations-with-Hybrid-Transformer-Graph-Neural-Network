
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, edge_dim=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W_node = nn.Linear(in_dim, out_dim)
        
        # Edge gate mechanism
        # Takes edge attributes and current edge state (if any)
        # We simplify: Edge gate uses: RelType(embedding), Attributes, Node features?
        # Paper: Gate = E_type + W_f*EdgeAttr + ...
        
        self.W_edge_attr = nn.Linear(edge_dim, out_dim)
        
        # Attention mechanism
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim)) # to mix edge info
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, D]
        # edge_index: [2, E]
        # edge_attr: [E, EdgeDim]
        
        N = x.size(0)
        E = edge_index.size(1)
        
        # Project nodes
        h = self.W_node(x).view(N, self.num_heads, self.head_dim) # [N, Heads, D_h]
        
        # Project edges (simple projection for now to match dim)
        # In full paper it's more complex with edge states update, we simplify for demo
        e = self.W_edge_attr(edge_attr).view(E, self.num_heads, self.head_dim) # [E, Heads, D_h]
        
        # Source and Target nodes for edges
        idx_src, idx_dst = edge_index
        h_src = h[idx_src] # [E, Heads, D_h]
        h_dst = h[idx_dst] # [E, Heads, D_h]
        
        # Attention scores
        # (h_src * att_src + h_dst * att_dst + e * att_edge)
        scores = (h_src * self.att_src).sum(dim=-1) + (h_dst * self.att_dst).sum(dim=-1) + (e * self.att_edge).sum(dim=-1)
        scores = self.leaky_relu(scores) # [E, Heads]
        
        # Softmax over neighbors
        # We need to map scores back to nodes to normalize
        # Using simple exp / sum_exp trick with scatter_add is tricky in raw PyTorch without torch_scatter
        # For simplicity, we can use a naive implementation or assume PyG logic if installed. 
        # But we act like we build from scratch.
        # We will use stable logs or simple softmax per node. 
        # Since standard torch doesn't have scatter_softmax efficiently, we cheat slightly by using unnormalized scores 
        # weighted by degree or use a simple loop (slow) or assume torch_geometric is available?
        # WAIT: The paper mentions specific GAT.
        # Let's try to do a manual softmax using index.
        
        # Exp
        scores = scores - scores.max()
        exp_scores = torch.exp(scores) # [E, Heads]
        
        # Sum exp per dst node
        sum_exp = torch.zeros(N, self.num_heads, device=x.device)
        sum_exp.index_add_(0, idx_dst, exp_scores)
        
        # Normalize
        # Add epsilon
        att_weights = exp_scores / (sum_exp[idx_dst] + 1e-6) # [E, Heads]
        
        # Aggregate
        # Weighted sum of neighbors (h_src)
        out_msg = h_src * att_weights.unsqueeze(-1) # [E, Heads, D_h]
        
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, idx_dst, out_msg) # [N, Heads, D_h]
        
        out = out.reshape(N, -1) # [N, OutDim]
        
        return out, edge_attr # We pass edge attr through (or update it if we implemented update)

class RelationalEncoder(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATLayer(in_dim, hidden_dim, num_heads=heads))
            in_dim = hidden_dim # Keeps dim same
            
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            out, _ = layer(x, edge_index, edge_attr)
            # Residual + Norm
            x = self.norm(x + out)
        return x
