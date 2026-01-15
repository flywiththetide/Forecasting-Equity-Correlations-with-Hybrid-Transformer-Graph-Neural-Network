
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_heatmap_comparison(pred_matrix, target_matrix, save_path, title_prefix=""):
    """
    Plot Heatmap of Predicted vs Actual Correlation Matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(target_matrix, ax=axes[0], cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    axes[0].set_title(f"{title_prefix} Actual Correlation")
    
    sns.heatmap(pred_matrix, ax=axes[1], cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    axes[1].set_title(f"{title_prefix} Predicted Correlation")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'heatmap_comparison.png'))
    plt.close()

def plot_residual_distribution(preds, targets, save_path):
    """
    Histogram of residuals (Pred - Actual).
    """
    residuals = preds - targets
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title("Distribution of Residuals")
    plt.xlabel("Prediction Error (Z-space)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_path, 'residual_distribution.png'))
    plt.close()

def plot_feature_importance(feature_names, importance_scores, save_path):
    """
    Bar chart of feature importance.
    """
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importance_scores, y=feature_names)
    plt.title("Feature Importance (Gradient x Input)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()
    
def plot_attention_map(att_weights, tickers, save_path):
    """
    Heatmap of attention weights for a sample node.
    att_weights: [N_neighbors]
    """
    # Assuming att_weights is a subgraph adj or similar
    # For demo we will plot random sample if full matrix not available
    pass

def plot_loss_by_sector(sectors, losses, save_path):
    """
    Bar chart of average loss per sector.
    """
    df = pd.DataFrame({'Sector': sectors, 'Loss': losses})
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Sector', y='Loss', ci=None)
    plt.xticks(rotation=45)
    plt.title("Average Loss by Sector")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_by_sector.png'))
    plt.close()

def plot_time_series_pair(dates, actuals, preds, pair_name, save_path):
    """
    Line chart of actual vs predicted correlation over time for a specific pair.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actuals, label='Actual')
    plt.plot(dates, preds, label='Predicted', linestyle='--')
    plt.title(f"Correlation Time Series: {pair_name}")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'time_series_pair.png'))
    plt.close()

def plot_rolling_mse(dates, mses, save_path):
    """
    Line chart of MSE over time.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, mses)
    plt.title("Rolling MSE over Time")
    plt.xlabel("Date")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'rolling_mse.png'))
    plt.close()

def plot_expert_usage(usage_counts, save_path):
    """
    Bar chart of how often High/Mid/Low experts are used.
    """
    labels = ['Negative', 'Neutral', 'Positive'] # 0, 1, 2
    plt.figure(figsize=(6, 6))
    plt.bar(labels, usage_counts)
    plt.title("Expert Head Usage Frequency")
    plt.savefig(os.path.join(save_path, 'expert_usage.png'))
    plt.close()

def plot_pred_vs_volatility(volatilities, errors, save_path):
    """
    Scatter plot of Stock Volatility vs Prediction Error.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(volatilities, errors, alpha=0.3)
    plt.xlabel("Stock Volatility (Annualized)")
    plt.ylabel("Mean Absolute Error")
    plt.title("Prediction Error vs Volatility")
    plt.savefig(os.path.join(save_path, 'error_vs_volatility.png'))
    plt.close()
    
def plot_degree_distribution(degrees, save_path):
    """
    Histogram of node degrees in the graph.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(degrees, bins=20)
    plt.title("Graph Degree Distribution")
    plt.xlabel("Degree")
    plt.savefig(os.path.join(save_path, 'degree_dist.png'))
    plt.close()
