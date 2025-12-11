
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader
from graphtorch.dataset import TorchDataset
from graphtorch.model import GraphTORCHModel
from tqdm import tqdm
import os
import numpy as np

def train():
    # Config
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    LR = 1e-3
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    # InMemoryDataset automatically calls process() if files are missing
    dataset = TorchDataset(root='data')
    
    # Validation split check    # Dataset
    # Assuming you want to load the processed data from 'data' dir (or specify filename if reprocessing needed)
    # If processing is already done (data.pt exists), filename arg is ignored by InMemoryDataset logic usually,
    # but we pass it just in case.
    dataset = TorchDataset(root='data', filename='data/gnn-inputs-train.root', treename='filteredHits')
    # Force process if needed or just load
    # dataset.process() 
    
    # Shuffle and Split
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset Size: {len(dataset)}. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = GraphTORCHModel(HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Handle Class Imbalance
    # Calculate ratio of neg/pos edges from a sample or estimate
    # We can iterate the training set once or just estimate. 
    # Let's verify quickly from the first batch.
    print("Calculating class weight from first batch...")
    sample_batch = next(iter(train_loader))
    num_pos = sample_batch['track', 'to', 'hit'].y.sum()
    num_neg = sample_batch['track', 'to', 'hit'].y.size(0) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Pos Weight: {pos_weight_val:.2f} (Num Pos: {num_pos}, Num Neg: {num_neg})")
    
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward
            logits = model(data)
            
            # Get Ground Truth
            # Uses labels generated from trackID matching in dataset.py
            labels = data['track', 'to', 'hit'].y
            
            loss = criterion(logits.view(-1), labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        
        # Accumulate for Sklearn
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                logits = model(data).view(-1)
                labels = data['track', 'to', 'hit'].y
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute Epoch Metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        avg_val_loss = val_loss / len(val_loader)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Purity: {precision:.4f} | Eff: {recall:.4f} | F1: {f1:.4f}")
        
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/graphtorch_model.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()
