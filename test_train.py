import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data.dataset import RobotCDFDataset
from utils.cdf_net import CDF_Net
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def forward_kinematics(q, link_lengths):
    x, y = 0, 0
    positions = [(x, y)]
    for i in range(len(link_lengths)):
        x += link_lengths[i] * np.cos(np.sum(q[:i+1]))
        y += link_lengths[i] * np.sin(np.sum(q[:i+1]))
        positions.append((x, y))
    return np.array(positions)

def sample_point_on_edge(start, end, t):
    return start + t * (end - start)

def generate_zero_config_dataset(link_lengths, num_configs, points_per_config):
    configurations = []
    points = []
    cdf_values = []
    
    for _ in tqdm(range(num_configs), desc="Generating configurations"):
        # Sample a random configuration
        q = np.random.uniform(-np.pi, np.pi, len(link_lengths))
        
        # Compute joint positions
        joint_positions = forward_kinematics(q, link_lengths)
        
        # Sample points on each link
        for i in range(len(link_lengths)):
            start = joint_positions[i]
            end = joint_positions[i+1]
            
            for _ in range(points_per_config // len(link_lengths)):
                t = np.random.random()
                point = sample_point_on_edge(start, end, t)
                
                configurations.append(q)
                points.append(point)
                cdf_values.append(0.0)  # CDF value is zero for all points
    
    return {
        'configurations': configurations,
        'points': points,
        'cdf_values': cdf_values
    }

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

def generate_random_inputs(batch_size, num_links, device):
    configs = torch.rand(batch_size, num_links, device=device) * 2 * np.pi - np.pi
    points = torch.rand(batch_size, 2, device=device) * 4 - 2  # Range [-2, 2]
    
    encoded_configs = torch.cat([configs, torch.sin(configs), torch.cos(configs)], dim=1)
    return torch.cat([encoded_configs, points], dim=1)

def train(model, dataloader, num_epochs, learning_rate, device, num_links, loss_threshold=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()
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
            outputs = model(inputs).squeeze()
            mse_loss = mse_criterion(outputs, targets)
            
            # Generate samples for eikonal loss
            configs = inputs[:, :num_links]
            points = inputs[:, -2:]
            random_inputs = sampler.get_samples(configs, points, batch_size, device)
            random_inputs.requires_grad = True
            
            # Forward pass for random inputs
            random_outputs = model(random_inputs).squeeze()
            
            # Compute eikonal loss
            grad_outputs = torch.ones_like(random_outputs)
            gradients = torch.autograd.grad(random_outputs, random_inputs, grad_outputs=grad_outputs, create_graph=True)[0]
            
            # Extract gradients for angles
            angle_gradients = gradients[:, :num_links]
            
            # Compute eikonal loss
            eikonal_loss = torch.mean((angle_gradients.norm(2, dim=1) - 1)**2)
            
            # Combine losses
            loss = mse_loss + 0.2 * eikonal_loss
            
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

def plot_cdf_field(model, device, link_lengths, obstacle, resolution=50):
    print("Plotting CDF field...")
    
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)

    Z = np.zeros_like(Theta1)
    total_points = resolution ** 2
    
    model.eval()
    with torch.no_grad():
        for idx, (i, j) in enumerate(np.ndindex(Theta1.shape)):
            if idx % 100 == 0:
                print(f"CDF computation: {idx}/{total_points} points processed")
            
            q = np.array([Theta1[i, j], Theta2[i, j]])
            
            # Compute end effector position
            positions = forward_kinematics(q, link_lengths)
            end_effector = positions[-1]
            
            # Prepare input for the model
            encoded_config = np.concatenate([q, np.sin(q), np.cos(q)])
            input_data = np.concatenate([encoded_config, obstacle[:2]])
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get model prediction
            Z[i, j] = model(input_tensor).item()

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(Theta1, Theta2, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='CDF Value')

    ax.set_xlabel('θ1')
    ax.set_ylabel('θ2')
    ax.set_title('CDF Field for 2D Robot Arm')

    # Plot the zero level set (you may need to adjust the level value)
    zero_level = ax.contour(Theta1, Theta2, Z, levels=[0.05], colors='r', linewidths=2)
    ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    plt.savefig('cdf_field.png')
    plt.close()

    print("CDF field plot saved as 'cdf_field.png'")

def main():
    use_generated_dataset = False  # Set to False to use the loaded dataset
    link_lengths = np.array([2.0, 2.0])  # Assuming 2 links of length 2

    if use_generated_dataset:
        # Generate the dataset
        num_configs = 400
        points_per_config = 1000
        data = generate_zero_config_dataset(link_lengths, num_configs, points_per_config)

        configurations = data['configurations']
        points = data['points']
        cdf_values = data['cdf_values']
    else:
        # Load the dataset (original code for loading)
        data = np.load('cdf_dataset/robot_cdf_dataset_2_links.npy', allow_pickle=True).item()
        configurations = data['configurations']
        points = data['points']
        cdf_values = data['cdf_values']

    print("\nDetailed Dataset info:")
    print(f"Configurations shape: {np.array(configurations).shape}")
    print(f"Points shape: {np.array(points).shape}")
    print(f"CDF values shape: {np.array(cdf_values).shape}")
    print(f"Number of configurations: {len(configurations)}")
    print(f"Number of points: {len(points)}")

    # Create dataset and dataloader
    dataset = RobotCDFDataset(configurations, points, cdf_values)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    num_links = len(configurations[0])
    input_dims = num_links * 3 + 2  # Changed to account for raw angles, sin, cos, and 2D point
    model = CDF_Net(input_dims=input_dims, hidden_dims=[512, 512, 512, 512], skip_in=[4], geometric_init=True).cuda()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    learning_rate = 0.0015
    loss_threshold = 1e-3  # Set your desired loss threshold here
    trained_model, best_loss = train(model, dataloader, num_epochs, learning_rate, device, num_links, loss_threshold)
    
    print(f"Training complete! Best loss: {best_loss:.4f}")
    
    # Plot CDF field
    obstacle = np.array([1.5, 1.5, 0])
    plot_cdf_field(trained_model, device, link_lengths, obstacle)

    torch.save(trained_model.state_dict(), f"trained_models/cdf_models/cdf_model_2_links.pt")
    print("Saved best trained CDF model")

if __name__ == "__main__":
    main()
