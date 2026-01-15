
import torch
import torch.optim as optim
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    
    # We assume dataloader yields (x, edge_index, edge_attr, target_z, base_z)
    # Since we built a custom structure, we might iterate manually or via a simple loader
    
    for batch in dataloader:
        x, edge_index, edge_attr, target_z, base_z, _ = batch
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        target_z = target_z.to(device)
        
        # Forward
        delta_z_pred = model(x, edge_index, edge_attr)
        
        # Prediction in Z space = base + delta
        z_pred = base_z.to(device) + delta_z_pred
        
        # Loss
        loss, l_edge, l_hist = criterion(z_pred, target_z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1

        
    return total_loss / count if count > 0 else 0

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, edge_attr, target_z, base_z, _ = batch
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            target_z = target_z.to(device)
            
            delta_z_pred = model(x, edge_index, edge_attr)
            z_pred = base_z.to(device) + delta_z_pred
            
            loss, _, _ = criterion(z_pred, target_z)
            total_loss += loss.item()
            count += 1
            
            preds.append(z_pred.cpu().numpy())
            targets.append(target_z.cpu().numpy())
            
    return total_loss / count if count > 0 else 0, preds, targets
