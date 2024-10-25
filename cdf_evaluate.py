import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.cdf_net import CDF_Net
import re
import jax
import jax.numpy as jnp


def forward_kinematics(q, link_lengths):
    x, y = 0, 0
    positions = [(x, y)]
    for i in range(len(link_lengths)):
        x += link_lengths[i] * np.cos(np.sum(q[:i+1]))
        y += link_lengths[i] * np.sin(np.sum(q[:i+1]))
        positions.append((x, y))
    return np.array(positions)

def load_learned_cdf(trained_model_path="trained_models/cdf_models/cdf_model_2_links.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_links = infer_num_links(trained_model_path)
    
    state_dict = torch.load(trained_model_path, map_location=device)
    model = CDF_Net(
        input_dims=num_links * 3 + 2,
        hidden_dims=[512, 512, 512, 512],
        output_dims=1,
        skip_in=(4,),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def cdf_evaluate_model(model, configs, points, device, batch_size=1024):
    model.eval()
    cdf_values = []
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
        
        # Process in batches
        num_samples = input_tensor.shape[0]
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_input = input_tensor[start:end]
            batch_cdf_values = model(batch_input.view(-1, batch_input.shape[-1]))
            cdf_values.append(batch_cdf_values.cpu().numpy())
    
    return np.concatenate(cdf_values, axis=0).reshape(configs_tensor.shape[0], points_tensor.shape[1])


def infer_num_links(trained_model_path):
    match = re.search(r'(\d+)_links', trained_model_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Number of links not found in the model path.")

def plot_cdf_field(ax, model, device, joint_pair, num_links, obstacle, resolution=50):
    theta = np.linspace(-np.pi, np.pi, resolution)
    Theta1, Theta2 = np.meshgrid(theta, theta)
    
    Z = np.zeros_like(Theta1)
    for idx, (x, y) in enumerate(np.ndindex(Theta1.shape)):
        q = np.zeros(num_links)
        q[joint_pair[0]] = Theta1[x, y]
        q[joint_pair[1]] = Theta2[x, y]
        # Set fixed angles for other joints
        fixed_joints = [j for j in range(num_links) if j not in joint_pair]
        q[fixed_joints] = np.pi / 4  # Example fixed angle

        # Evaluate CDF for the current configuration
        Z[x, y] = cdf_evaluate_model(model, q[np.newaxis, :], obstacle[:2].reshape(1, -1), device)

    # Plot CDF
    contour = ax.contourf(Theta1, Theta2, Z, levels=20, cmap='viridis')
    zero_level = ax.contour(Theta1, Theta2, Z, levels=[0.1], colors='r', linewidths=2)
    plt.colorbar(contour, ax=ax, label='CDF Value')

    # Add labels to the zero level set
    ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    ax.set_xlabel(f'θ{joint_pair[0] + 1}')
    ax.set_ylabel(f'θ{joint_pair[1] + 1}')
    ax.set_title(f'CDF Field for Joint Pair {joint_pair}')


def main():
    trained_model_path = "trained_models/cdf_models/cdf_model_4_links.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_learned_cdf(trained_model_path).to(device)

    obstacle = np.array([1.5, 1.5, 0])  # Example obstacle position

    num_links = infer_num_links(trained_model_path)
    joint_pairs = [(i, j) for i in range(num_links) for j in range(i + 1, num_links)]
    num_plots = len(joint_pairs)

    # Calculate grid dimensions
    cols = 2
    rows = (num_plots + cols - 1) // cols  # Ceiling division to determine rows

    # Set figsize to ensure square subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axs = axs.ravel()

    for i, joint_pair in enumerate(joint_pairs):
        plot_cdf_field(axs[i], model, device, joint_pair, num_links, obstacle)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig('cdf_field.png')
    plt.close()

    print("CDF field plot saved as 'cdf_field.png'")

if __name__ == "__main__":
    main()




