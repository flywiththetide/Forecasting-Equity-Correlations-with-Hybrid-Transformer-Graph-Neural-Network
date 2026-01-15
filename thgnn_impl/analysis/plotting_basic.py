
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def plot_scatter(preds, targets, save_path, title='Prediction Scatter'):
    preds_flat = np.concatenate(preds)
    targets_flat = np.concatenate(targets)
    
    if len(preds_flat) > 10000:
        idx = np.random.choice(len(preds_flat), 10000, replace=False)
        preds_flat = preds_flat[idx]
        targets_flat = targets_flat[idx]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(targets_flat, preds_flat, alpha=0.1, s=1)
    # y=x line
    mi = min(targets_flat.min(), preds_flat.min())
    ma = max(targets_flat.max(), preds_flat.max())
    plt.plot([mi, ma], [mi, ma], 'r--')
    plt.xlabel('Actual Z-Correlation')
    plt.ylabel('Predicted Z-Correlation')
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'scatter_pred.png'))
    plt.close()
