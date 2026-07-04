"""
train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from quantum_model import DynamicHybridQNN
from utils import load_dataset, set_seed, create_directories, save_model

def preprocess_data(images, labels):
    # Flatten images from [N, 4, 4] to [N, 16]
    X = images.reshape(images.shape[0], -1).astype(np.float32)
    # Convert pixels to FRQI angles: (pixel / 255) * (pi / 2)
    X_angles = (X / 255.0) * (np.pi / 2.0)
    
    # Qiskit TorchConnector outputs shape [Batch, 1], so labels should match
    y = labels.astype(np.float32).reshape(-1, 1)
    
    return torch.tensor(X_angles), torch.tensor(y)

def train():
    set_seed(42)
    create_directories()
    
    # Load raw data
    X_train_raw, y_train_raw = load_dataset("dataset/train")
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
    
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize dynamic model (4x4 images = 16 pixels)
    model = DynamicHybridQNN(image_size=4, reps=2)
    
    # Loss & Optimizer
    # QNN with Z-measurement outputs values between [-1, 1]. 
    # Use MSELoss to regress towards -1 (class 0) and 1 (class 1).
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    print("Starting training...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Map labels from [0, 1] to targets [-1, 1] to match expectation values
            targets = batch_y * 2.0 - 1.0
            
            outputs = model(batch_x)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(train_loader.dataset):.4f}")
        
    save_model(model, "quantum_polyp_model.pt")
    print("Model saved safely.")

if __name__ == "__main__":
    train()