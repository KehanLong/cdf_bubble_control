import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from arm_2d_utils import forward_kinematics, transform_shape, generate_robot_point_cloud
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

def generate_workspace_points(num_points, radius=8):
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


def compute_cdf_values_with_surface_points(configs, points, params_list, num_points_per_link=200):
    """Compute CDF values for all configurations and points, including surface points."""
    
    # Compute CDF values for workspace points
    workspace_cdf_values = compute_cdf_values(configs, points, params_list)
    
    # Generate surface points for all configurations
    surface_points_list = [generate_robot_point_cloud(config, num_points_per_link) for config in configs]
    
    # All configurations have the same number of surface points
    num_surface_points = len(surface_points_list[0])
    total_points = len(points) + num_surface_points
    
    # Prepare arrays for all points and CDF values
    all_points = np.zeros((len(configs), total_points, 2))
    cdf_values = np.zeros((len(configs), total_points))
    
    for i, surface_points in enumerate(surface_points_list):
        # Combine workspace points and surface points
        all_points[i] = np.vstack([points, surface_points])
        
        # Store CDF values (workspace CDF values + zeros for surface points)
        cdf_values[i, :len(points)] = workspace_cdf_values[i]
        # Surface points already have CDF value 0, so we don't need to set them explicitly
    
    return all_points, cdf_values

def visualize_dataset_sample(configs, all_points, cdf_values, sample_idx, save_path='dataset_sample_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Get the configuration and all points for this sample
    config = configs[sample_idx]
    points = all_points[sample_idx]
    
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
        params = jnp.load(f"trained_models/sdf_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)

    # Generate data
    num_configs = 100
    num_points = 1000
    num_surface_points_link = 100

    configs = generate_robot_configurations(num_configs)
    if NUM_LINKS == 2:
        radius = 8
    if NUM_LINKS == 5:
        radius = 19
    workspace_points = generate_workspace_points(num_points, radius)
    all_points, cdf_values = compute_cdf_values_with_surface_points(configs, workspace_points, params_list, num_points_per_link=num_surface_points_link)

    # Save data
    os.makedirs('cdf_dataset', exist_ok=True)
    data = {
        'configurations': configs,
        'points': all_points,
        'cdf_values': cdf_values
    }
    np.save('cdf_dataset/robot_cdf_dataset_2_links.npy', data)

    print("Data generation complete. File saved as 'cdf_dataset/robot_cdf_dataset_2_links.npy'.")

    # Visualize a sample from the dataset
    random_sample_idx = np.random.randint(0, num_configs)
    visualize_dataset_sample(configs, all_points, cdf_values, sample_idx=random_sample_idx, save_path='cdf_dataset/dataset_sample_visualization_2_links.png')

if __name__ == "__main__":
    main()