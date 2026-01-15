
import torch
import torch.nn as nn
from .transformer import TemporalEncoder
from .gat import RelationalEncoder
from .experts import ExpertHead

class THGNN(nn.Module):
    def __init__(self, num_features, seq_len=30, d_model=128, gat_dim=512):
        super().__init__()
        
        self.temporal = TemporalEncoder(num_features, d_model=d_model, seq_len=seq_len)
        self.structural = RelationalEncoder(in_dim=512, hidden_dim=gat_dim)
        
        # Input: Concat(h_i, h_j, e_ij) -> 512 + 512 + EdgeAttrDim(6) ~= 1030
        self.expert_input_dim = 512 + 512 + 6 
        
        self.expert_neg = ExpertHead(self.expert_input_dim)
        self.expert_mid = ExpertHead(self.expert_input_dim)
        self.expert_pos = ExpertHead(self.expert_input_dim)
        
    def forward(self, x, edge_index, edge_attr):
        h_nodes = self.temporal(x) 
        h_ctx = self.structural(h_nodes, edge_index, edge_attr) 
        
        src, dst = edge_index
        h_src = h_ctx[src]
        h_dst = h_ctx[dst]
        
        u_ij = torch.cat([h_src, h_dst, edge_attr], dim=1) 
        rel_types = edge_attr[:, -1].long()
        
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
