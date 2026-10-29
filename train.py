"""
train.py
Updated to support explicit profiling tracking and runtime arguments.
"""

import argparse
import time
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from quantum_model import DynamicHybridQNN
from utils import load_dataset, set_seed, create_directories, save_model

def preprocess_data(images, labels, encoding_type):
    X = images.reshape(images.shape[0], -1).astype(np.float32)
    
    # Standardize scale transformation bounds depending on the targets
    if encoding_type in ["frqi", "neqr"]:
        X_encoded = (X / 255.0) * (np.pi / 2.0)
    else: 
        # Amplitude and QWT accept normalized continuous bounds directly
        X_encoded = X / 255.0
        
    y = labels.astype(np.float32).reshape(-1, 1)
    return torch.tensor(X_encoded), torch.tensor(y)

def train():
    parser = argparse.ArgumentParser(description="Hybrid QNN Training Platform Engine")
    parser.add_argument(
        "--encoding", 
        type=str, 
        default="frqi", 
        choices=["frqi", "amplitude", "neqr", "qwt"],
        help="Select the quantum image state preparation layer protocol (default: frqi)"
    )
    args = parser.parse_args()

    set_seed(22)
    create_directories()
    
    print(f"Initializing dataset loading using configuration space: [Encoding: {args.encoding.upper()}]")
    X_train_raw, y_train_raw = load_dataset("dataset/train")
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw, args.encoding)
    
    total_images = len(X_train)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize dynamic model passing custom encoding option parameter
    model = DynamicHybridQNN(image_size=8, reps=2, encoding=args.encoding)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    print("Starting training process profiling...")
    epochs = 10
    
    # Global Profiling Clock Started
    start_total_time = time.perf_counter()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.perf_counter()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            targets = batch_y * 2.0 - 1.0
            outputs = model(batch_x)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_image_time_epoch = (epoch_duration / total_images) * 1000  # in milliseconds

        print(f"Epoch {epoch+1:02d}/{epochs} "
              f"- Loss: {epoch_loss / total_images:.4f} "
              f"- Time: {epoch_duration:.2f}s "
              f"- Avg Speed: {avg_image_time_epoch:.2f} ms/image")
        
    # Global Profiling Clock Ended
    end_total_time = time.perf_counter()
    total_training_duration = end_total_time - start_total_time
    grand_avg_image_time = (total_training_duration / (total_images * epochs)) * 1000

    print("\n" + "="*50)
    print(" TRAINING PROFILE COMPLETE STATISTICS")
    print("="*50)
    print(f"Selected Encoding Method : {args.encoding.upper()}")
    print(f"Total Dataset Images     : {total_images}")
    print(f"Total Training Time      : {total_training_duration:.2f} seconds")
    print(f"Grand Average Velocity   : {grand_avg_image_time:.3f} ms / image-pass")
    print("="*50 + "\n")
        
    save_model(model, f"quantum_polyp_model_{args.encoding}.pt")
    print("Model variant checkpoint saved safely.")

    subprocess.run([sys.executable, "evaluate.py", "--encoding", args.encoding])

if __name__ == "__main__":
    train()