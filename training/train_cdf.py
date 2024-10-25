import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn

class CDFImprovedSampler:
    def __init__(self, num_links, global_sigma=np.pi, local_sigma_config=0.1, local_sigma_point=0.1):
        self.num_links = num_links
        self.workspace_radius = num_links * 2
        self.global_sigma = global_sigma
        self.local_sigma_config = local_sigma_config
        self.local_sigma_point = local_sigma_point

    def get_samples(self, configs, points, batch_size, device):
        # Local samples (close to zero-config points)
        local_configs = configs + torch.randn_like(configs) * self.local_sigma_config
        local_points = points + torch.randn_like(points) * self.local_sigma_point
        
        # Global samples
        global_configs = torch.rand(batch_size // 2, self.num_links, device=device) * 2 * np.pi - np.pi
        global_points = self.sample_disk_points(batch_size // 2, device)
        
        # Combine local and global samples
        combined_configs = torch.cat([local_configs, global_configs], dim=0)
        combined_points = torch.cat([local_points, global_points], dim=0)
        
        # Encode configurations
        encoded_configs = torch.cat([combined_configs, torch.sin(combined_configs), torch.cos(combined_configs)], dim=1)
        
        # Combine encoded configs and points
        samples = torch.cat([encoded_configs, combined_points], dim=1)
        
        return samples

    def sample_disk_points(self, num_points, device):
        r = torch.sqrt(torch.rand(num_points, device=device)) * self.workspace_radius
        theta = torch.rand(num_points, device=device) * 2 * np.pi
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=1)

def generate_random_configs(batch_size, num_links, device):
    return torch.rand(batch_size, num_links, device=device) * 2 * np.pi - np.pi

def generate_random_points(batch_size, workspace_radius, device):
    r = torch.sqrt(torch.rand(batch_size, device=device)) * workspace_radius
    theta = torch.rand(batch_size, device=device) * 2 * np.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack((x, y), dim=1)

def train(model, dataloader, num_epochs=500, learning_rate=0.001, device='cuda', use_improved_sampler=False, loss_threshold=0.01):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()
    
    # Infer num_links from the first batch of the dataloader
    first_batch = next(iter(dataloader))
    inputs, _ = first_batch
    num_links = (inputs.shape[1] - 2) // 3  # Subtract 2 for the 2D point, then divide by 3

    workspace_radius = num_links * 2

    # Initialize the improved sampler if needed
    if use_improved_sampler:
        sampler = CDFImprovedSampler(num_links)

    best_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_eikonal_loss = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]
            
            optimizer.zero_grad()
            
            # Forward pass for dataset samples
            outputs = model(inputs).squeeze(-1)
            mse_loss = mse_criterion(outputs, targets)
            
            # Generate random inputs for eikonal loss
            if use_improved_sampler:
                random_inputs = sampler.get_samples(inputs[:, :num_links], inputs[:, -2:], batch_size, device)
            else:
                random_configs = generate_random_configs(batch_size, num_links, device)
                random_points = generate_random_points(batch_size, workspace_radius, device)
                random_inputs = torch.cat((random_configs, torch.sin(random_configs), torch.cos(random_configs), random_points), dim=1)
            
            random_inputs.requires_grad = True
            
            # Forward pass for random inputs
            random_outputs = model(random_inputs).squeeze(-1)
            
            # Compute eikonal loss
            grad_outputs = torch.ones_like(random_outputs)
            gradients = torch.autograd.grad(random_outputs, random_inputs, grad_outputs=grad_outputs, create_graph=True)[0]
            
            # Extract gradients for angles
            angle_gradients = gradients[:, :num_links]
            
            # Compute eikonal loss
            eikonal_loss = torch.mean((angle_gradients.norm(2, dim=1) - 1)**2)
            
            # Combine losses
            loss = mse_loss + 0.1 * eikonal_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_eikonal_loss += eikonal_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mse_loss = total_mse_loss / len(dataloader)
        avg_eikonal_loss = total_eikonal_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, Eikonal Loss: {avg_eikonal_loss:.4f}")

        # Check if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
        
        # Check if we've reached the loss threshold
        if avg_loss <= loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break

    # Load the best model state
    model.load_state_dict(best_model_state)
    return model, best_loss
