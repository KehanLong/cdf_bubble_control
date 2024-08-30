import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from arm_2d_utils import forward_kinematics, transform_shape
from arm_2d_cdf import find_closest_zero_config
from data.arm_2d_config import shapes, NUM_LINKS
import os

from tqdm import tqdm


def generate_robot_configurations(num_configs):
    """Generate random robot configurations."""
    configs = []
    for _ in range(num_configs):
        config = np.random.uniform(-np.pi, np.pi, 1).tolist() + \
                 np.random.uniform(-np.pi/2, np.pi/2, NUM_LINKS-1).tolist()
        configs.append(config)
    return np.array(configs)

def generate_workspace_points(num_points, radius=19):
    """Generate random points within a disk."""
    r = radius * np.sqrt(np.random.random(num_points))
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def compute_cdf_values(configs, points, params_list):
    """Compute CDF values for all configurations and points using full batch processing."""
    cdf_values = np.zeros((len(configs), len(points)))
    
    for j, point in tqdm(enumerate(points), desc="Processing points", total=len(points)):
        _, batch_cdf_values, _ = find_closest_zero_config(point, configs, params_list)
        cdf_values[:, j] = batch_cdf_values
    
    return cdf_values

def visualize_dataset_sample(configs, points, cdf_values, sample_idx, save_path='dataset_sample_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Get the configuration and all points for this sample
    config = configs[sample_idx]
    
    # Create scatter plot
    scatter = ax.scatter(points[:, 0], points[:, 1], c=cdf_values[sample_idx], cmap='viridis', s=10, vmin=0, vmax=2)
    
    # Plot robot arm
    joint_positions = forward_kinematics(config)
    current_angle = 0
    
    for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
        if i < len(config):
            current_angle += config[i]
        transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
        ax.fill(*zip(*transformed_shape), alpha=0.5)
        ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title(f'CDF Visualization for Robot Arm (Sample {sample_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('CDF Value')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")

def main():
    # Load parameters
    params_list = []
    for i in range(NUM_LINKS):
        params = jnp.load(f"trained_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)

    # Generate data
    num_configs = 150
    num_points = 2000

    configs = generate_robot_configurations(num_configs)
    points = generate_workspace_points(num_points)
    cdf_values = compute_cdf_values(configs, points, params_list)

    # Save data
    os.makedirs('cdf_dataset', exist_ok=True)
    data = {
        'configurations': configs,
        'points': points,
        'cdf_values': cdf_values
    }
    np.save('cdf_dataset/robot_cdf_dataset.npy', data)

    print("Data generation complete. File saved as 'cdf_dataset/robot_cdf_dataset.npy'.")

    # Visualize a sample from the dataset
    random_sample_idx = np.random.randint(0, num_configs)
    visualize_dataset_sample(configs, points, cdf_values, sample_idx=random_sample_idx, save_path='cdf_dataset/dataset_sample_visualization.png')


if __name__ == "__main__":
    main()