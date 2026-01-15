
import torch

def compute_gradient_input(model, x, edge_index, edge_attr, target_idx=None):
    """
    Computes Gradient x Input for feature importance.
    """
    x.requires_grad = True
    output = model(x, edge_index, edge_attr)
    
    if target_idx is not None:
        score = output[target_idx]
    else:
        score = output.mean()
        
    score.backward()
    
    # Grad * Input
    importance = x.grad * x
    importance = importance.mean(dim=(0, 1)) # Avg over batch/seq
    
    return importance.detach().cpu().numpy()
