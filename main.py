
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from thgnn_impl import config
from thgnn_impl.data.fetcher import fetch_and_process_data
from thgnn_impl.data.dataset import GraphDataset
from thgnn_impl.models.thgnn import THGNN
from thgnn_impl.training.losses import HybridLoss
from thgnn_impl.training.trainer import train_one_epoch, validate
from thgnn_impl.analysis import plotting_basic, plotting_advanced, metrics, interpretation
from thgnn_impl.utils.logging import setup_logger
from thgnn_impl.utils.helpers import set_seed, count_parameters

def main():
    # Setup
    os.makedirs('results', exist_ok=True)
    logger = setup_logger('results')
    set_seed(42)
    device = config.DEVICE
    logger.info(f"Running on {device}")
    
    # 1. Load Data
    aligned_data = fetch_and_process_data(config.TICKERS, config.START_DATE, config.END_DATE)
    tickers = config.TICKERS
    
    # Create Dataset Wrapper
    dataset_wrapper = GraphDataset(aligned_data, tickers, seq_len=config.SEQ_LEN)
    
    # Split
    total_days = dataset_wrapper.total_days
    train_size = int(total_days * 0.8)
    
    logger.info("Preparing Train Data...")
    train_data = dataset_wrapper.get_data_range(0, train_size)
    logger.info("Preparing Val Data...")
    val_data = dataset_wrapper.get_data_range(train_size, total_days)
    
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # 2. Model
    num_feats = aligned_data[tickers[0]].shape[1]
    
    model = THGNN(num_features=num_feats, seq_len=config.SEQ_LEN, d_model=config.D_MODEL).to(device)
    logger.info(f"Model parameters: {count_parameters(model)}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = HybridLoss().to(device)
    
    # 3. Train
    train_losses = []
    val_losses = []
    
    logger.info("Starting Training...")
    epochs = config.EPOCHS
    
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        t_loss = train_one_epoch(model, train_data, criterion, optimizer, device)
        v_loss, val_preds_list, val_targets_list = validate(model, val_data, criterion, device)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
    # 4. Results & Analysis
    logger.info("Generating Analysis...")
    plotting_basic.plot_loss(train_losses, val_losses, 'results')
    
    # Get final validation outputs (concatenated)
    # v_loss, val_preds_list, val_targets_list were from last epoch
    plotting_basic.plot_scatter(val_preds_list, val_targets_list, 'results')
    
    val_preds = np.concatenate(val_preds_list)
    val_targets = np.concatenate(val_targets_list)
    
    # Metrics
    met = metrics.compute_metrics(val_preds, val_targets)
    logger.info(f"Final Metrics: {met}")
    
    # Generate 10 New Graphs
    
    # 1. Heatmap Comparison (Last batch sample)
    # We take the last sample from val_data
    # Re-run forward on last sample
    last_sample = val_data[-1]
    x, edge_index, edge_attr, target_z, base_z, time_idx = last_sample
    with torch.no_grad():
        delta = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        pred_z = base_z.to(device) + delta
    
    # Reconstruct Matrix (Sparse to Dense) - Just fill edges for heatmap
    # N x N matrix
    N = len(tickers)
    pred_matrix = np.zeros((N, N))
    target_matrix = np.zeros((N, N))
    
    src, dst = edge_index.cpu().numpy()
    pred_vals = pred_z.cpu().numpy()
    target_vals = target_z.cpu().numpy()
    
    pred_matrix[src, dst] = pred_vals
    target_matrix[src, dst] = target_vals
    
    plotting_advanced.plot_heatmap_comparison(pred_matrix, target_matrix, 'results')
    
    # 2. Residual Distribution
    plotting_advanced.plot_residual_distribution(val_preds, val_targets, 'results')
    
    # 3. Feature Importance (Gradient x Input on last sample)
    imp_scores = interpretation.compute_gradient_input(model, x.to(device), edge_index.to(device), edge_attr.to(device))
    feat_names = aligned_data[tickers[0]].columns
    # Ensure lengths match (imp_scores matches num features)
    if len(imp_scores) == len(feat_names):
        plotting_advanced.plot_feature_importance(feat_names, imp_scores, 'results')
        
    # 4. Expert Usage
    # Count edge types in validation set
    total_edges = 0
    neg_edges = 0
    mid_edges = 0
    pos_edges = 0
    for batch in val_data:
        _, _, edge_attr, _, _, _ = batch
        # Last col is type
        types = edge_attr[:, -1]
        neg_edges += (types == 0).sum().item()
        mid_edges += (types == 1).sum().item()
        pos_edges += (types == 2).sum().item()
    
    plotting_advanced.plot_expert_usage([neg_edges, mid_edges, pos_edges], 'results')
    
    # 5. Rolling MSE
    # We can use val_losses as proxy if we tracked per day, but val_losses is avg per epoch.
    # Let's compute day-wise error from val_data
    day_errors = []
    day_indices = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_data:
            x, edge_index, edge_attr, target_z, base_z, t_idx = batch
            delta = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            pred = base_z.to(device) + delta
            mse = torch.mean((pred - target_z.to(device))**2).item()
            day_errors.append(mse)
            day_indices.append(t_idx)
            
    plotting_advanced.plot_rolling_mse(day_indices, day_errors, 'results')
    
    # 6. Degree Distribution
    deg = np.bincount(src)
    plotting_advanced.plot_degree_distribution(deg, 'results')
    
    # 7. Time Series Pair (First pair in graph)
    # Track one edge over time
    # This requires consistent edge indexing which isn't guaranteed by build_graph dynamic.
    # We skip for now or do "Avg Correlation" over time?
    # Let's plot "Avg Predicted Correlation" vs "Avg Actual" over time
    avg_pred = []
    avg_act = []
    
    with torch.no_grad():
        for batch in val_data:
            x, edge_index, edge_attr, target_z, base_z, _ = batch
            delta = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            pred = base_z.to(device) + delta
            avg_pred.append(pred.mean().item())
            avg_act.append(target_z.to(device).mean().item())
            
    plotting_advanced.plot_time_series_pair(day_indices, avg_act, avg_pred, "Average Market Correlation", 'results')
    
    # 8. Loss by Stock (Sector Proxy)
    # We don't have sector mapping loaded easily here. 
    # Let's do Volatility vs Error
    # Volatility is feature 8 ('vol_20'). 
    # Get mean volatility for graph
    vols = []
    errs = []
    for i, batch in enumerate(val_data):
        x = batch[0] # [30, N, F]
        # Mean volatility of all stocks in this day
        # F=8 is vol_20? Check features. 
        # features list: [Close, Vol, ret, log, m5, m20, m60, rev5, vol20...]
        # vol_20 is index 8.
        curr_vol = x[:, :, 8].mean().item()
        vols.append(curr_vol)
        errs.append(day_errors[i])
        
    plotting_advanced.plot_pred_vs_volatility(vols, errs, 'results')
        
    logger.info("Done! Results saved to /results")

if __name__ == "__main__":
    main()
