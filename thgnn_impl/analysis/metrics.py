
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(preds, targets):
    """
    Computes MAE, RMSE, and Correlation.
    preds, targets: 1D numpy arrays
    """
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    
    # Correlation
    if len(preds) > 1:
        corr = np.corrcoef(preds, targets)[0, 1]
    else:
        corr = 0.0
        
    return {
        'mae': mae,
        'rmse': rmse,
        'corr': corr
    }
