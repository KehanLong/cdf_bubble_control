import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.cdf_net import CDF_Net
import re


def forward_kinematics(q, link_lengths):
    x, y = 0, 0
    positions = [(x, y)]
    for i in range(len(link_lengths)):
        x += link_lengths[i] * np.cos(np.sum(q[:i+1]))
        y += link_lengths[i] * np.sin(np.sum(q[:i+1]))
        positions.append((x, y))
    return np.array(positions)

def load_learned_cdf(trained_model_path="trained_models/cdf_models/cdf_model_1_links.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_links = infer_num_links(trained_model_path)
    
    state_dict = torch.load(trained_model_path, map_location=device)
    model = CDF_Net(
        input_dims=num_links * 3 + 2,
        output_dims=1,
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
        if points_tensor.dim() == 3:  # If already [B, N, 2]
            points_tensor = points_tensor.squeeze(0)  # Remove batch dim if present
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
    # ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    ax.set_xlabel(f'θ{joint_pair[0] + 1}')
    ax.set_ylabel(f'θ{joint_pair[1] + 1}')
    ax.set_title(f'CDF Field for Joint Pair {joint_pair}')


def plot_cdf_comparison_1link(model, device, resolution=100, fixed_angle=-0.0, link_length=2.0):
    """
    Create 2D plots comparing true and predicted CDF values across the workspace.
    Fixed robot angle at 0 (pointing right).
    """
    # Create coordinate grid
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create mask for points within reach
    distances = np.sqrt(X**2 + Y**2)
    mask = distances <= link_length
    
    # Initialize arrays for CDF values
    Z_network = np.full_like(X, np.nan)
    Z_true = np.full_like(X, np.nan)
    
    # Compute CDF values for points within reach
    points = []
    for i in range(resolution):
        for j in range(resolution):
            if mask[i, j]:
                points.append([X[i, j], Y[i, j]])
    
    if points:
        points = np.array(points)
        config = np.array([fixed_angle])
        
        # Get network predictions and compute norm
        distances = cdf_evaluate_model(model, config, points, device).flatten()
        network_cdfs = distances  # Already computed as norm in cdf_evaluate_model
        
        # Compute true CDFs
        point_angles = np.arctan2(points[:, 1], points[:, 0])
        true_cdfs = np.abs(np.arctan2(np.sin(fixed_angle - point_angles), 
                                     np.cos(fixed_angle - point_angles)))
        
        # Fill in the values
        idx = 0
        for i in range(resolution):
            for j in range(resolution):
                if mask[i, j]:
                    Z_network[i, j] = network_cdfs[idx]
                    Z_true[i, j] = true_cdfs[idx]
                    idx += 1
    
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot network predictions
    im1 = ax1.contourf(X, Y, Z_network, levels=20, cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='CDF Value')
    ax1.set_title('Network Predictions')
    
    # Plot true CDF values
    im2 = ax2.contourf(X, Y, Z_true, levels=20, cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='CDF Value')
    ax2.set_title('True CDF Values')
    
    # Plot difference
    diff = Z_network - Z_true
    im3 = ax3.contourf(X, Y, diff, levels=20, cmap='RdBu', center=0)
    plt.colorbar(im3, ax=ax3, label='Difference')
    ax3.set_title('Difference (Network - True)')
    
    # Add robot arm visualization to all plots
    arm_x = [0, link_length * np.cos(fixed_angle)]
    arm_y = [0, link_length * np.sin(fixed_angle)]
    for ax in [ax1, ax2, ax3]:
        ax.plot(arm_x, arm_y, 'r-', linewidth=2, label='Robot Arm')
        ax.scatter(0, 0, color='black', s=100, label='Base')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('cdf_comparison_1link.png')
    plt.close()

def check_network_gradients(model, device, num_samples=10, link_length=2.0):
    """
    Check the gradients and divergence of the network with respect to input angle.
    """
    print("\nChecking Network Gradients and Divergence:")
    print("-" * 100)
    print(f"{'Point':20} {'Angle':10} {'CDF Value':12} {'|Gradient|':12} {'True CDF':12} {'Divergence':12}")
    print("-" * 100)
    
    for i in range(num_samples):
        # Create angle with gradient tracking
        angle = torch.tensor([[np.random.uniform(-np.pi, np.pi)]], 
                           requires_grad=True, device=device)
        
        # Sample point
        r = np.random.uniform(0, link_length)
        theta = np.random.uniform(0, 2*np.pi)
        point = torch.tensor([[r * np.cos(theta), r * np.sin(theta)]], 
                           dtype=torch.float32, device=device)
        
        # Create network input
        network_input = torch.cat([
            angle,
            torch.sin(angle),
            torch.cos(angle),
            point
        ], dim=1)
        
        # Forward pass - now returns per-joint distances
        distances = model(network_input)
        # Compute CDF as norm of distances
        cdf_value = torch.norm(distances, dim=-1, keepdim=True)
        
        # Compute first derivative (gradient)
        grad_outputs = torch.ones_like(cdf_value)
        grad = torch.autograd.grad(cdf_value, angle, 
                                 grad_outputs=grad_outputs,
                                 create_graph=True)[0]
        
        # Compute second derivative (divergence)
        div = torch.autograd.grad(grad, angle,
                                grad_outputs=torch.ones_like(grad),
                                create_graph=False)[0]
        
        # Compute true CDF
        point_angle = np.arctan2(point[0, 1].item(), point[0, 0].item())
        true_cdf = np.abs(np.arctan2(np.sin(angle.item() - point_angle), 
                                    np.cos(angle.item() - point_angle)))
        
        point_str = f"({point[0,0].item():.3f}, {point[0,1].item():.3f})"
        print(f"{point_str:10} {angle.item():10.3f} "
              f"{cdf_value.item():12.3f} {abs(grad.item()):12.3f} "
              f"{true_cdf:12.3f} {div.item():12.3f}")
    
    print("-" * 100)

def test_specific_cases(model, device):
    """Test network outputs for specific configurations and points."""
    print("\nTesting specific cases:")
    print("-" * 50)
    
    test_cases = [
        {
            'config': [-1.38, -3.0],
            'point': [-0.1, 0.55],
            'description': "Test case 1"
        },
        {
            'config': [0.46, -1.69],
            'point': [-1.15, -0.75],
            'description': "Test case 2"
        },
        {
            'config': [2.56, -2.88],
            'point': [0.9, -0.6],
            'description': "Test case 3"
        },
        # Add more test cases as needed
    ]
    
    for case in test_cases:
        config = np.array(case['config'])
        point = np.array(case['point'])
        
        # Get network prediction
        cdf_value = cdf_evaluate_model(
            model, 
            config[np.newaxis, :],  # Add batch dimension
            point[np.newaxis, :],   # Add batch dimension
            device
        )
        
        print(f"\n{case['description']}:")
        print(f"Configuration: [{config[0]:.3f}, {config[1]:.3f}]")
        print(f"Point: [{point[0]:.3f}, {point[1]:.3f}]")
        print(f"CDF Value: {cdf_value[0][0]:.4f}")
        print("-" * 50)

def main():
    trained_model_path = "trained_models/cdf_models/cdf_model_2_links_truncated_new.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_links = infer_num_links(trained_model_path)

    model = load_learned_cdf(trained_model_path).to(device)
    
    
    if num_links == 1:
        print("Detected 1-link model, checking gradients...")
        check_network_gradients(model, device)
        
        print("\nGenerating comparison visualization...")
        plot_cdf_comparison_1link(model, device)
        print("Comparison plot saved as 'cdf_comparison_1link.png'")
    else:
        print(f"Detected {num_links}-link model, generating CDF field plot...")
        # Test specific cases
        test_specific_cases(model, device)
        obstacle = np.array([-0.1, 0.5, 0])  # Example obstacle position
        
        # For 2-link robot, we only have one joint pair
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_cdf_field(ax, model, device, (0, 1), num_links, obstacle)

        plt.tight_layout()
        plt.savefig('cdf_field.png')
        plt.close()

        print("CDF field plot saved as 'cdf_field.png'")

if __name__ == "__main__":
    main()




