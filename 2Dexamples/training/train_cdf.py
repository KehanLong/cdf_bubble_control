import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

def check_gradients(model, num_links, workspace_radius, device, num_samples=100):
    """
    Check the actual gradient norms for random samples
    """
    model.eval()
    
    # Generate random samples with gradient tracking
    random_angles = generate_random_configs(num_samples, num_links, device)
    random_angles.requires_grad = True  # Enable gradient tracking on angles
    random_points = generate_random_points(num_samples, workspace_radius, device)
    
    # Create full input maintaining computational graph
    random_inputs = torch.cat((
        random_angles, 
        torch.sin(random_angles), 
        torch.cos(random_angles), 
        random_points
    ), dim=1)
    
    # Forward pass
    outputs = model(random_inputs).squeeze(-1)
    
    # Compute gradients w.r.t. angles directly
    grad_outputs = torch.ones_like(outputs)
    angle_gradients = torch.autograd.grad(outputs, random_angles, grad_outputs=grad_outputs)[0]
    
    gradient_norms = angle_gradients.norm(2, dim=1)
    
    # Compute statistics
    avg_norm = gradient_norms.mean().item()
    std_norm = gradient_norms.std().item()
    max_norm = gradient_norms.max().item()
    min_norm = gradient_norms.min().item()
    
    print(f"\nGradient Statistics:")
    print(f"Average gradient norm: {avg_norm:.4f}")
    print(f"Std of gradient norms: {std_norm:.4f}")
    print(f"Max gradient norm: {max_norm:.4f}")
    print(f"Min gradient norm: {min_norm:.4f}")
    
    # Add histogram-like statistics
    num_below_threshold = torch.sum(gradient_norms < 0.9).item()
    num_above_threshold = torch.sum(gradient_norms > 1.1).item()
    
    print(f"Gradients < 0.9: {num_below_threshold}/{num_samples} ({100*num_below_threshold/num_samples:.1f}%)")
    print(f"Gradients > 1.1: {num_above_threshold}/{num_samples} ({100*num_above_threshold/num_samples:.1f}%)")
    
    return avg_norm

def get_divergence_weight(epoch, total_epochs):
    """
    Returns divergence weight based on training phase:
    - First 50% epochs: High divergence (weight = 1.0)
    - Next 25% epochs: Linear annealing from 1.0 to 0.1
    - Final 25% epochs: Low divergence (weight = 0.1)
    """
    half_point = total_epochs // 2
    three_quarter_point = 3 * total_epochs // 4
    
    if epoch < half_point:  # High divergence phase
        return 1.0
    elif epoch < three_quarter_point:  # Annealing phase
        progress = (epoch - half_point) / (three_quarter_point - half_point)
        return 1.0 - 0.9 * progress  # Linear interpolation from 1.0 to 0.1
    else:  # Low divergence phase
        return 0.1




def train(model, dataloader, num_epochs=500, learning_rate=0.001, device='cuda', 
          use_improved_sampler=False, loss_threshold=0.01, training_mode='truncated'):
    """
    Args:
        training_mode (str): Either 'binary' for zero/truncated/off-robot or 'continuous' for pre-computed CDFs
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # Option 1: StepLR - Decays the learning rate by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Option 2: ReduceLROnPlateau - Reduces learning rate when metric has stopped improving
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=20, 
    #     verbose=True, min_lr=1e-6
    # )
    
    # Option 3: CosineAnnealingLR - Cycles the learning rate with cosine annealing
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs, eta_min=1e-6
    # )
    
    # Option 4: OneCycleLR - Implements the 1cycle policy
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=learning_rate, epochs=num_epochs, 
    #     steps_per_epoch=len(dataloader)
    # )

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
        total_eikonal_loss_inputs = 0
        total_off_manifold_loss = 0
        total_laplacian_loss = 0
        

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]
            
            optimizer.zero_grad()
            
            # Enable gradient tracking for input configurations
            input_configs = inputs[:, :num_links].detach().clone()
            input_configs.requires_grad = True
            
            # Reconstruct full input tensor with gradient tracking
            full_inputs = torch.cat((
                input_configs,
                torch.sin(input_configs),
                torch.cos(input_configs),
                inputs[:, -2:]  # points
            ), dim=1)
            
            # Forward pass for dataset samples
            outputs = model(full_inputs).squeeze(-1)

            # Compute losses based on training mode
            if training_mode == 'truncated':
                # Split between off-robot points and all other points (including truncated)
                off_robot_mask = targets == 100

                # MSE loss for all points except off-robot points
                mse_mask = ~off_robot_mask
                if outputs[mse_mask].numel() > 0:
                    mse_loss = F.mse_loss(outputs[mse_mask], targets[mse_mask])
                else:
                    mse_loss = torch.tensor(0.0, device=device)

                # Off-manifold loss for points not on the robot
                off_manifold_pred = outputs[off_robot_mask]
                if off_manifold_pred.numel() > 0:
                    off_manifold_loss = torch.exp(-1e2 * off_manifold_pred).mean()
                else:
                    off_manifold_loss = torch.tensor(0.0, device=device)

            else:  # 'continuous' mode
                # Direct MSE loss for all points as they have pre-computed CDF values
                mse_loss = F.mse_loss(outputs, targets)
                off_manifold_loss = torch.tensor(0.0, device=device)

            # Compute eikonal loss for input configurations
            grad_outputs_inputs = torch.ones_like(outputs)
            grad_cdf_inputs = torch.autograd.grad(
                outputs=outputs,  # Already scalar
                inputs=input_configs,
                grad_outputs=grad_outputs_inputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            eikonal_loss_inputs = torch.mean((grad_cdf_inputs.norm(2, dim=1) - 1).square())
            
            # Generate random inputs for eikonal loss
            if use_improved_sampler:
                random_inputs = sampler.get_samples(inputs[:, :num_links], inputs[:, -2:], batch_size, device)
            else:
                random_configs = generate_random_configs(batch_size, num_links, device)
                random_configs.requires_grad = True  # Enable gradient tracking here
                random_points = generate_random_points(batch_size, workspace_radius, device)
                random_inputs = torch.cat((random_configs, torch.sin(random_configs), torch.cos(random_configs), random_points), dim=1)
            
            # Forward pass for random inputs
            random_outputs = model(random_inputs).squeeze(-1)  # Now [batch_size]
            
            # Compute gradient of scalar CDF w.r.t random_configs
            grad_outputs = torch.ones_like(random_outputs)
            grad_cdf = torch.autograd.grad(
                outputs=random_outputs,  # Already scalar
                inputs=random_configs,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute Eikonal loss
            eikonal_loss = torch.mean((grad_cdf.norm(2, dim=1) - 1).square())
            
            # Compute divergence (sum of second derivatives)
            laplacian = torch.zeros(random_configs.shape[0], device=device)
            for i in range(num_links):
                grad_cdf_i = grad_cdf[:, i]  # Shape: [batch_size]
                
                # Compute second derivative for each sample
                second_deriv = torch.autograd.grad(
                    grad_cdf_i,
                    random_configs,
                    grad_outputs=torch.ones_like(grad_cdf_i),
                    create_graph=True,
                    retain_graph=True
                )[0][:, i]  # Shape: [batch_size]
                
                laplacian += second_deriv
            laplacian_loss = torch.mean(laplacian ** 2)


            
            
            # Combine losses based on training mode
            if training_mode == 'truncated':
                #loss = 3.0 * mse_loss + 0.05 * eikonal_loss + 0.1 * off_manifold_loss + 0.01 * laplacian_loss
                loss = 3.0 * mse_loss + 0.05 * eikonal_loss + 0.1 * off_manifold_loss
            else:  # 'continuous' mode
                loss = mse_loss + 0.04 * eikonal_loss 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
  
            optimizer.step()
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_eikonal_loss += eikonal_loss.item()
            total_eikonal_loss_inputs += eikonal_loss_inputs.item()
            total_off_manifold_loss += off_manifold_loss.item()
            total_laplacian_loss += laplacian_loss.item()


        avg_loss = total_loss / len(dataloader)
        avg_mse_loss = total_mse_loss / len(dataloader)
        avg_eikonal_loss = total_eikonal_loss / len(dataloader)
        avg_eikonal_loss_inputs = total_eikonal_loss_inputs / len(dataloader)
        avg_off_manifold_loss = total_off_manifold_loss / len(dataloader)
        avg_laplacian_loss = total_laplacian_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, "
              f"MSE Loss: {avg_mse_loss:.4f}, "
              f"Eikonal Loss: {avg_eikonal_loss:.4f}, "
              f"eikonal loss inputs: {avg_eikonal_loss_inputs:.4f}, "
              f"off manifold Loss: {avg_off_manifold_loss:.4f}, "
              f"laplacian Loss: {avg_laplacian_loss:.4f}")
        
        
        # Check if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
        
        # Check if we've reached the loss threshold
        if avg_loss <= loss_threshold:
            print(f"Loss threshold {loss_threshold} reached. Stopping training.")
            break

        # Update learning rate based on scheduler type
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)  # For ReduceLROnPlateau
        else:
            scheduler.step()  # For other schedulers
        
        # Print current learning rate along with other metrics
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch [{epoch+1}/{num_epochs}], "
        #       f"Learning Rate: {current_lr:.6f}, "
        #       )

    # Load the best model state
    model.load_state_dict(best_model_state)
    return model, best_loss

