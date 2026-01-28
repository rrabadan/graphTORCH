
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
from graphtorch.dataset import TorchDataset
from graphtorch.model import GraphTORCHModel
from tqdm import tqdm
import os
import numpy as np

def train(input_file, epochs, batch_size=64, hidden_size=64, lr=1e-3, root_dir='data'):
    # Config
    BATCH_SIZE = batch_size
    HIDDEN_SIZE = hidden_size
    LR = lr
    EPOCHS = epochs
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    # Assuming loading the processed data from 'data' dir (or specify filename if reprocessing needed)
    # If processing is already done (data.pt exists), filename arg is ignored by InMemoryDataset logic usually,
    # but pass it just in case.
    dataset = TorchDataset(root=root_dir, filename=input_file, treename='events')
    # Force process if needed or just load
    # dataset.process() 
    
    # Shuffle and Split
    torch.manual_seed(42)
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
    # Verify quickly from the first batch.
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
        all_probs = []
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
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute Epoch Metrics
        all_preds = np.concatenate(all_preds) # This is 0.5 threshold
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        avg_val_loss = val_loss / len(val_loader)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"Epoch {epoch+1} | Loss: {avg_val_loss:.4f} | AUC: {auc:.4f}")
        
        # Check multiple thresholds
        thresholds = [0.5, 0.8, 0.95]
        for thresh in thresholds:
            preds_t = (all_probs > thresh).astype(float)
            purity = precision_score(all_labels, preds_t, zero_division=0)
            eff = recall_score(all_labels, preds_t, zero_division=0)
            f1 = f1_score(all_labels, preds_t, zero_division=0)
            print(f"  > Thresh {thresh:.2f} | Purity: {purity:.4f} | Eff: {eff:.4f} | F1: {f1:.4f}")
        
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/graphtorch_model.pth')
    print("Model saved.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GraphTORCH Model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input ROOT file')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--root_dir', type=str, default='data', help='Data root directory (default: data)')
    
    args = parser.parse_args()
    
    train(input_file=args.input_file, 
          epochs=args.epochs,
          batch_size=args.batch_size,
          hidden_size=args.hidden_size,
          lr=args.lr,
          root_dir=args.root_dir)

if __name__ == "__main__":
    main()
