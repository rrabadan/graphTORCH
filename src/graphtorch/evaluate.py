import torch
import torch.nn as nn # For BCEWithLogitsLoss if needed, though we just need model
from torch_geometric.loader import DataLoader
from graphtorch.dataset import TorchDataset
from graphtorch.model import GraphTORCHModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np
import argparse
import os

def evaluate(input_file, model_path, root_dir='data', batch_size=64, hidden_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Dataset ---
    # Uses the same parameters as training to ensure the same data structure
    print(f"Loading dataset from {input_file}...")
    dataset = TorchDataset(root=root_dir, filename=input_file, treename='events')
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # To make it reproducible:
    # Ideally we should save the train/val split indices or seed to ensure we evaluate on validation set specifically.
    torch.manual_seed(42)
    dataset = dataset.shuffle() 
    train_size = int(0.8 * len(dataset))
    dataset = dataset[train_size:]
    
    # --- 2. Load Model ---
    print(f"Loading model from {model_path}...")
    model = GraphTORCHModel(hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- 3. Inference ---
    all_probs = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data).view(-1)
            probs = torch.sigmoid(logits)
            labels = data['track', 'to', 'hit'].y
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # --- 4. Compute ROC ---
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # --- 5. Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Efficiency)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    output_png = 'roc_curve.png'
    plt.savefig(output_png)
    print(f"ROC curve saved to {os.path.abspath(output_png)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphTORCH Model and Plot ROC')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input ROOT file')
    parser.add_argument('--model_path', type=str, default='models/graphtorch_model.pth', help='Path to saved model .pth file')
    parser.add_argument('--root_dir', type=str, default='data', help='Data root directory')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size used in training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    
    args = parser.parse_args()
    
    evaluate(args.input_file, args.model_path, args.root_dir, args.batch_size, args.hidden_size)

if __name__ == "__main__":
    main()
