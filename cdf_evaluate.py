import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.cdf_net import CDF_Net
from data.arm_2d_config import NUM_LINKS


def forward_kinematics(q, link_lengths):
    x, y = 0, 0
    positions = [(x, y)]
    for i in range(len(link_lengths)):
        x += link_lengths[i] * np.cos(np.sum(q[:i+1]))
        y += link_lengths[i] * np.sin(np.sum(q[:i+1]))
        positions.append((x, y))
    return np.array(positions)

def load_learned_cdf(trained_model_path="trained_models/cdf_models/cdf_model_zeroconfigs_2_links_best.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dict = torch.load(trained_model_path, map_location=device)
    model = CDF_Net(
        input_dims=NUM_LINKS * 3 + 2,
        hidden_dims=[512, 512, 512, 512],
        output_dims=1,
        skip_in=(4,),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def cdf_evaluate_model(model, configs, points, device):
    model.eval()
    with torch.no_grad():
        # Ensure configs and points are 2D
        if configs.ndim == 1:
            configs = configs[np.newaxis, :]
        if points.ndim == 1:
            points = points[np.newaxis, :]
        
        # Encode configurations
        encoded_configs = np.concatenate([configs, np.sin(configs), np.cos(configs)], axis=-1)
        
        # Convert to tensors and move to device
        configs_tensor = torch.tensor(encoded_configs, dtype=torch.float32).to(device)
        points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
        
        # Reshape for broadcasting
        configs_tensor = configs_tensor.unsqueeze(1)  # (N, 1, config_dim)
        points_tensor = points_tensor.unsqueeze(0)    # (1, M, point_dim)
        
        # Prepare input for the model
        input_tensor = torch.cat([
            configs_tensor.expand(-1, points_tensor.shape[1], -1),
            points_tensor.expand(configs_tensor.shape[0], -1, -1)
        ], dim=-1)
        
        # Get model predictions
        cdf_values = model(input_tensor.view(-1, input_tensor.shape[-1]))
        cdf_values = cdf_values.view(configs_tensor.shape[0], points_tensor.shape[1])
    
    return cdf_values.cpu().numpy()

def plot_cdf_field(model, device, link_lengths, obstacle, resolution=50, save_path='cdf_field.png'):
    print("Plotting CDF field...")
    
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)

    configs = np.stack([Theta1.ravel(), Theta2.ravel()], axis=1)
    obstacle_point = obstacle[:2].reshape(1, -1)  # Use only x, y coordinates, ensure 2D

    # Evaluate CDF for all configurations at once
    Z = cdf_evaluate_model(model, configs, obstacle_point, device)
    Z = Z.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(Theta1, Theta2, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='CDF Value')

    ax.set_xlabel('θ1')
    ax.set_ylabel('θ2')
    ax.set_title('CDF Field for 2D Robot Arm')

    # Plot the zero level set (you may need to adjust the level value)
    zero_level = ax.contour(Theta1, Theta2, Z, levels=[0.1], colors='r', linewidths=2)
    ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    plt.savefig(save_path)
    plt.close()

    print(f"CDF field plot saved as '{save_path}'")

def main():
    trained_model_path = "trained_models/cdf_models/cdf_model_zeroconfigs_2_links_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_learned_cdf(trained_model_path).to(device)

    link_lengths = np.array([2.0, 2.0])  # Assuming 2 links of length 2
    obstacle = np.array([1.5, 1.5, 0])  # Example obstacle position

    plot_cdf_field(model, device, link_lengths, obstacle)

if __name__ == "__main__":
    main()
