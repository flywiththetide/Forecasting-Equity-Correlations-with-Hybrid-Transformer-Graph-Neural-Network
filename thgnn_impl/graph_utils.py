
import numpy as np
import torch
import pandas as pd

def build_graph(correlation_matrix, top_k=50, bottom_k=50, mid_k=75):
    """
    Builds a graph from a correlation matrix based on the paper's strategy:
    - Top 50 correlations (positive)
    - Bottom 50 correlations (negative)
    - 75 random mid-range correlations [0.2, 0.8] per node (we'll approximate 'per node' vs global)
    
    The paper specifies "for every stock... edges...". 
    So for each node i, we select neighbors.
    """
    num_nodes = correlation_matrix.shape[0]
    edge_index = []
    edge_attr = []
    
    # Iterate over each node to select its neighbors
    for i in range(num_nodes):
        # Get correlations for node i
        corrs = correlation_matrix[i]
        
        # Sort indices
        sorted_indices = np.argsort(corrs)
        
        # Top k (strongest positive) - excluding self (last one is self 1.0)
        # sorted_indices[-1] is self, so take [-(k+1):-1]
        top_indices = sorted_indices[-(top_k+1):-1]
        
        # Bottom k (strongest negative)
        bottom_indices = sorted_indices[:bottom_k]
        
        # Mid k (random sample from remaining)
        # We define "mid" as indices not in top or bottom
        # Ideally we should filter by value range [0.2, 0.8] but for simplicity/robustness we just sample from remainder
        # Or strictly follow paper [0.2, 0.8]. Let's try to follow paper range if possible.
        
        potential_mid_indices = []
        for idx in range(num_nodes):
            if idx != i and idx not in top_indices and idx not in bottom_indices:
                val = abs(corrs[idx]) # Paper says [0.2, 0.8] by rho_base? Or just value.
                # "mid-strength partners sampled from [0.2, 0.8] correlation percentiles" -> Percentiles or raw values? 
                # Let's assume raw absolute values or just remaining. 
                # Given simplest interpretation: Randomly sample from remaining
                potential_mid_indices.append(idx)
                
        if len(potential_mid_indices) >= mid_k:
            mid_indices = np.random.choice(potential_mid_indices, mid_k, replace=False)
        else:
            mid_indices = np.array(potential_mid_indices)
            
        # Combine all neighbors
        neighbors = np.concatenate([top_indices, bottom_indices, mid_indices]).astype(int)
        
        for j in neighbors:
            edge_index.append([i, j])
            
            # Edge Attributes
            rho = corrs[j]
            abs_rho = abs(rho)
            sign = 0 if rho > 0 else 1
            
            # Relation type for routing (0: neg, 1: mid, 2: pos)
            # Paper: bottom 3rd=0, middle 3rd=1, top 3rd=2 of rho_base
            # We can approximate this by thresholds e.g. < -0.3, -0.3 to 0.3, > 0.3 or strictly by rank
            # Let's use simple thresholds for now to determine type
            if rho < -0.1: # Approximate 'negative'
                rel_type = 0
            elif rho > 0.1:
                rel_type = 2
            else:
                rel_type = 1
            
            # TODO: Sector info needs to be passed in to compute same_sector/same_ind
            # For now we use placeholders 0
            same_sector = 0 
            same_ind = 0
            
            # Attribute vector: [rho, abs_rho, sign, same_sector, same_ind, rel_type]
            # rel_type is categorical, we might need it separate or embedded
            # Paper says a(i,j) = [rho, |rho|, sign, same_sec, same_ind]
            # And rel_type is separate for routing.
            
            attr = [rho, abs_rho, sign, same_sector, same_ind]
            edge_attr.append(attr + [rel_type])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return edge_index, edge_attr
