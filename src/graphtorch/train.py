
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from graphtorch.dataset import TorchDataset
from graphtorch.model import GraphTORCHModel
from tqdm import tqdm
import os

def train():
    # Config
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    LR = 1e-3
    EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    # InMemoryDataset automatically calls process() if files are missing
    dataset = TorchDataset(root='data')
    
    # Validation split check (optional)
    print(f"Dataset loaded with {len(dataset)} events.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = GraphTORCHModel(HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for data in tqdm(loader, desc=f"Epoch {epoch+1}"):
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
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/graphtorch_model.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()
