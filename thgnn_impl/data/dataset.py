
import torch
import numpy as np
import pandas as pd
from .graph_builder import build_graph

class GraphDataset:
    def __init__(self, aligned_data, tickers, seq_len=30, pred_len=10):
        self.aligned_data = aligned_data
        self.tickers = tickers
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Prepare tensor
        data_arrays = []
        for t in tickers:
            data_arrays.append(aligned_data[t].values)
        
        # [Days, N, F]
        self.full_tensor = np.stack(data_arrays, axis=1)
        self.total_days = self.full_tensor.shape[0]
        
    def get_data_range(self, start_idx, end_idx):
        dataset = []
        print(f"Preparing dataset from idx {start_idx} to {end_idx}...")
        
        for t in range(start_idx, end_idx):
            if t < self.seq_len + self.pred_len: continue 
            
            # Input window
            x_window = self.full_tensor[t-self.seq_len:t] # [30, N, F]
            x_window = np.transpose(x_window, (1, 0, 2)) # [N, 30, F]
            
            # Correlation base (feature index 2 is approx returns or we recompute)
            # HACK: Using feature index 2 ('returns' in original features check)
            rets_window = x_window[:, :, 2]
            
            df_rets = pd.DataFrame(rets_window, columns=self.tickers)
            corr_matrix = df_rets.corr().fillna(0).values
            
            # Build Graph
            edge_index, edge_attr = build_graph(corr_matrix)
            
            # Target
            future_window = self.full_tensor[t:t+self.pred_len, :, 2]
            if future_window.shape[0] < self.pred_len: continue
            
            df_fut = pd.DataFrame(future_window, columns=self.tickers)
            target_corr = df_fut.corr().fillna(0).values
            
            # Edges
            src, dst = edge_index
            target_vals = target_corr[src, dst]
            base_vals = corr_matrix[src, dst]
            
            # Fisher Z safely
            target_z = np.arctanh(np.clip(target_vals, -0.99, 0.99))
            base_z = np.arctanh(np.clip(base_vals, -0.99, 0.99))
            
            dataset.append((
                torch.tensor(x_window, dtype=torch.float32),
                edge_index,
                edge_attr,
                torch.tensor(target_z, dtype=torch.float32),
                torch.tensor(base_z, dtype=torch.float32),
                t # Return time index for analysis
            ))
            
        return dataset
