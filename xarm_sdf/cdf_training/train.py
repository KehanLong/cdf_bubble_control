import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
import random

from network import CDFNetwork
from losses import compute_total_loss
from dataset import RobotCDFDataset

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_cdf_network(
    train_data_path,
    model_save_dir,
    batch_size=256,
    num_epochs=500,
    learning_rate=0.001,
    device='cuda',
    loss_threshold=0.001,
    seed=42
):
    # Set random seed
    set_seed(seed)
    
    # Convert paths to absolute paths
    train_data_path = Path(train_data_path)
    if not train_data_path.is_absolute():
        train_data_path = PROJECT_ROOT / train_data_path
    
    model_save_dir = Path(model_save_dir)
    if not model_save_dir.is_absolute():
        model_save_dir = PROJECT_ROOT / model_save_dir
    
    # Create save directory
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training data from: {train_data_path}")
    print(f"Saving models to: {model_save_dir}")
    
    # Load training data
    training_data = np.load(train_data_path, allow_pickle=True)
    configs = np.stack([item['joint_angles'] for item in training_data])
    points = np.stack([item['point'] for item in training_data])
    cdf_values = np.stack([item['cdf_value'] for item in training_data])
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Number of training pairs: {len(training_data)}")
    print(f"CDF value range: [{cdf_values.min():.4f}, {cdf_values.max():.4f}]")
    print(f"CDF mean: {cdf_values.mean():.4f}")
    print(f"CDF std: {cdf_values.std():.4f}")
    
    # Create dataset and dataloader
    dataset = RobotCDFDataset(configs, points, cdf_values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and move to device
    model = CDFNetwork().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_eikonal = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Compute all losses
            loss, mse, eikonal = compute_total_loss(model, inputs, targets, eikonal_weight=0.03)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_eikonal += eikonal.item()
        
        # Compute average losses
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_eikonal = total_eikonal / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f} MSE: {avg_mse:.6f} Eikonal: {avg_eikonal:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(model_save_dir, 'best_model_all_data.pth'))
        
        # Early stopping
        if avg_loss <= loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_loss

if __name__ == "__main__":
    # Use relative paths from project root
    train_data_path = "data/cdf_data/cdf_training_data_new_large.npy"
    model_save_dir = "trained_models/cdf"
    
    model, final_loss = train_cdf_network(
        train_data_path=train_data_path,
        model_save_dir=model_save_dir,
        batch_size=256,
        num_epochs=5000,
        learning_rate=2e-4,
        device='cuda',
        loss_threshold=1e-4,
        seed=42
    ) 