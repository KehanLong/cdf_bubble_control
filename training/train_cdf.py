import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from data.arm_2d_config import NUM_LINKS
import torch.nn as nn

def generate_random_configs(batch_size, num_links, device):
    return torch.rand(batch_size, num_links, device=device) * 2 * np.pi - np.pi

def generate_random_points(batch_size, workspace_radius, device):
    r = torch.sqrt(torch.rand(batch_size, device=device)) * workspace_radius
    theta = torch.rand(batch_size, device=device) * 2 * np.pi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack((x, y), dim=1)

def train(model, dataset, num_epochs=500, batch_size=128, learning_rate=0.001, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mse_criterion = nn.MSELoss()
    
    workspace_radius = NUM_LINKS * 2

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
            random_configs = generate_random_configs(batch_size, NUM_LINKS, device)
            random_points = generate_random_points(batch_size, workspace_radius, device)
            random_inputs = torch.cat((random_configs, torch.sin(random_configs), torch.cos(random_configs), random_points), dim=1)
            random_inputs.requires_grad = True
            
            # Forward pass for random inputs
            random_outputs = model(random_inputs).squeeze(-1)
            
            # Compute eikonal loss
            grad_outputs = torch.ones_like(random_outputs)
            gradients = torch.autograd.grad(random_outputs, random_inputs, grad_outputs=grad_outputs, create_graph=True)[0]
            
            # Extract gradients for angles
            angle_gradients = gradients[:, :NUM_LINKS]
            
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

    return model
