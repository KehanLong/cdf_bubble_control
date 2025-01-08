import torch
import torch.optim as optim
import numpy as np
import sys
import random
from pathlib import Path
from network import CDFNetwork
from losses import compute_total_loss

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.xarm_model import XArmFK

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CDFTrainer:
    def __init__(self, contact_db_path, device='cuda'):
        # Load compressed contact database
        self.db = np.load(contact_db_path, allow_pickle=True).item()
        self.valid_points = torch.tensor(self.db['points'], device=device)
        self.contact_configs = self.db['contact_configs']
        
        print(f"Loaded contact database with {len(self.valid_points)} points")
        print(f"Average configs per point: {np.mean([len(configs) for configs in self.contact_configs]):.1f}")
        
        # Initialize robot model for joint limits
        self.robot_fk = XArmFK(device=device)
        self.q_max = self.robot_fk.joint_limits[:, 1]
        self.q_min = self.robot_fk.joint_limits[:, 0]
        
        self.device = device
        self.batch_points = 200  
        self.batch_configs = 10  

    def sample_batch(self):
        """Sample batch of points and configs, compute CDF values online"""
        # 1. Sample random points from valid points
        point_indices = torch.randint(0, len(self.valid_points), (self.batch_points,))
        points = self.valid_points[point_indices]
        
        # 2. Sample random configurations
        configs = torch.rand(self.batch_configs, 6, device=self.device)
        configs = configs * (self.q_max - self.q_min) + self.q_min
        configs.requires_grad_(True)
        
        # 3. Compute CDF values efficiently
        cdf_values = torch.zeros(self.batch_points, self.batch_configs, device=self.device)
        for i, idx in enumerate(point_indices):
            contact_configs = torch.tensor(
                self.contact_configs[idx], 
                device=self.device
            )
            # Compute distances to all contact configurations at once
            distances = torch.cdist(
                configs,                    # [batch_configs, 6]
                contact_configs,            # [M_i, 6]
                p=2                         # L2 norm
            )
            cdf_values[i] = distances.min(dim=1)[0]
        
        return points, configs, cdf_values

def train_cdf_network(
    contact_db_path,
    model_save_dir,
    num_epochs=50000,
    learning_rate=0.001,
    device='cuda',
    loss_threshold=0.001
):
    # Convert paths to absolute paths
    contact_db_path = Path(contact_db_path)
    if not contact_db_path.is_absolute():
        contact_db_path = PROJECT_ROOT / contact_db_path
    
    model_save_dir = Path(model_save_dir)
    if not model_save_dir.is_absolute():
        model_save_dir = PROJECT_ROOT / model_save_dir
    
    # Create save directory
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading contact database from: {contact_db_path}")
    print(f"Saving models to: {model_save_dir}")
    
    # Initialize trainer
    trainer = CDFTrainer(contact_db_path, device)
    
    # Initialize model and optimizer
    model = CDFNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000,
        threshold=0.01, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        
        # Sample batch and compute CDF values
        points, configs, cdf_values = trainer.sample_batch()
        
        # Prepare inputs (all combinations of points and configs)
        points_exp = points.unsqueeze(1).expand(-1, trainer.batch_configs, -1)
        configs_exp = configs.unsqueeze(0).expand(trainer.batch_points, -1, -1)
        
        inputs = torch.cat([
            configs_exp.reshape(-1, 6),  # Configs first
            points_exp.reshape(-1, 3)    # Points last
        ], dim=1)
        
        targets = cdf_values.reshape(-1)
        
        # Forward pass and loss computation using helper
        optimizer.zero_grad()
        loss, mse_loss, eikonal_loss = compute_total_loss(
            model=model,
            inputs=inputs,
            targets=targets,
            eikonal_weight=0.04
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Update scheduler
        scheduler.step(loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.6f} MSE: {mse_loss.item():.6f} Eikonal: {eikonal_loss.item():.6f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_dir / 'best_model.pth')
        
        # Early stopping
        if loss.item() <= loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_loss

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed(42)  # You can change this seed value
    
    # Use relative paths from project root
    contact_db_path = "data/cdf_data/contact_db.npy"
    model_save_dir = "trained_models/cdf"
    
    model, final_loss = train_cdf_network(
        contact_db_path=contact_db_path,
        model_save_dir=model_save_dir,
        num_epochs=20000,
        learning_rate=0.001,
        device='cuda',
        loss_threshold=1e-4
    ) 