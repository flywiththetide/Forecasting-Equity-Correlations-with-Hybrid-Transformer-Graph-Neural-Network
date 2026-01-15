
import torch
import torch.nn as nn
from .transformer import TemporalEncoder
from .gat import RelationalEncoder

class THGNN(nn.Module):
    def __init__(self, num_features, seq_len=30, d_model=128, gat_dim=512):
        super().__init__()
        
        # 1. Temporal Encoder
        self.temporal = TemporalEncoder(num_features, d_model=d_model, seq_len=seq_len)
        
        # 2. Relational Encoder
        self.structural = RelationalEncoder(in_dim=512, hidden_dim=gat_dim)
        
        # 3. Expert Heads
        # Input: Concat(h_i, h_j, e_ij) -> 512 + 512 + EdgeAttrDim(6) ~= 1030
        self.expert_input_dim = 512 + 512 + 6 
        
        self.expert_neg = self._build_mlp()
        self.expert_mid = self._build_mlp()
        self.expert_pos = self._build_mlp()
        
    def _build_mlp(self):
        return nn.Sequential(
            nn.Linear(self.expert_input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1) # Output delta_z
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, Seq, Feat]
        edge_index: [2, E]
        edge_attr: [E, 6] (Where 6 is avg attr size, last one is rel_type)
        """
        
        # 1. Get Node Embeddings
        h_nodes = self.temporal(x) # [N, 512]
        
        # 2. Get Updated Contextual Nodes
        h_ctx = self.structural(h_nodes, edge_index, edge_attr) # [N, 512]
        
        # 3. Pairwise Prediction
        src, dst = edge_index
        h_src = h_ctx[src]
        h_dst = h_ctx[dst]
        
        # Concat for edge embedding
        # edge_attr includes [rho, abs, sign, sec, ind, type]
        # We include raw attributes in prediction input
        u_ij = torch.cat([h_src, h_dst, edge_attr], dim=1) # [E, 1030]
        
        # Routing by type (last col of edge_attr)
        # 0: neg, 1: mid, 2: pos
        rel_types = edge_attr[:, -1].long()
        
        # Predict
        delta_z = torch.zeros(u_ij.size(0), 1, device=x.device)
        
        mask_neg = (rel_types == 0)
        mask_mid = (rel_types == 1)
        mask_pos = (rel_types == 2)
        
        if mask_neg.any():
            delta_z[mask_neg] = self.expert_neg(u_ij[mask_neg])
        if mask_mid.any():
            delta_z[mask_mid] = self.expert_mid(u_ij[mask_mid])
        if mask_pos.any():
            delta_z[mask_pos] = self.expert_pos(u_ij[mask_pos])
            
        return delta_z.squeeze()
