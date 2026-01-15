
import torch
import torch.nn as nn
import torch.nn.functional as F

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)
        
    def forward(self, pred, target):
        return self.loss(pred, target)

class HistogramLoss(nn.Module):
    def __init__(self, num_bins=15, min_val=-1.0, max_val=1.0, sigma=0.1):
        super().__init__()
        # Create bin centers
        self.num_bins = num_bins
        self.step = (max_val - min_val) / (num_bins - 1)
        self.register_buffer('centers', torch.linspace(min_val, max_val, num_bins))
        self.sigma = sigma
        
    def forward(self, pred, target):
        """
        pred: [N] predicted values
        target: [N] target values
        """
        # Soft binning using Gaussian Kernel
        # Expand pred to [N, Bins]
        pred_ex = pred.unsqueeze(1) # N x 1
        target_ex = target.unsqueeze(1) # N x 1
        centers = self.centers.unsqueeze(0) # 1 x Bins
        
        # Gaussian waits
        w_pred = torch.exp( - (pred_ex - centers)**2 / (2 * self.sigma**2) )
        w_target = torch.exp( - (target_ex - centers)**2 / (2 * self.sigma**2) )
        
        # Normalize to get histogram
        h_pred = torch.mean(w_pred, dim=0) # [Bins]
        h_target = torch.mean(w_target, dim=0) # [Bins]
        
        # MSE between histograms
        loss = F.mse_loss(h_pred, h_target)
        return loss

class HybridLoss(nn.Module):
    def __init__(self, edge_weight=1.0, hist_weight=1.0, num_bins=15):
        super().__init__()
        self.edge_loss = HuberLoss()
        self.hist_loss = HistogramLoss(num_bins=num_bins)
        self.edge_weight = edge_weight
        self.hist_weight = hist_weight
        
    def forward(self, pred, target):
        l_edge = self.edge_loss(pred, target)
        l_hist = self.hist_loss(pred, target)
        return self.edge_weight * l_edge + self.hist_weight * l_hist, l_edge, l_hist
