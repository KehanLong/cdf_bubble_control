import numpy as np
import jax
import jax.numpy as jnp
import time
from arm_2d_cdf_compute import SimpleCDF2D, precompute_configs_jax
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import Normalize

@jax.jit
def compute_cdf_batch(configs, zero_configs, touching_indices):
    def partial_distance(config, zero_config, touch_index):
        # Create a mask based on the touch_index
        mask = jnp.arange(config.shape[0]) <= touch_index
        # Compute the difference and apply the mask
        diff = (config - zero_config) * mask
        return jnp.linalg.norm(diff)
    
    distances = jax.vmap(lambda c: jax.vmap(partial_distance, (None, 0, 0))(c, zero_configs, touching_indices))(configs)
    return jnp.min(distances, axis=1)

class CDFDatasetGenerator:
    def __init__(self, link_lengths, num_points, num_configs, num_zero_configs):
        self.cdf = SimpleCDF2D(link_lengths)
        self.num_points = num_points
        self.num_configs = num_configs
        self.num_zero_configs = num_zero_configs
        self.workspace_radius = np.sum(link_lengths)

    def generate_workspace_points(self):
        angles = np.random.uniform(0, 2*np.pi, self.num_points)
        radii = np.random.uniform(0, self.workspace_radius, self.num_points)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def generate_configs(self):
        return np.random.uniform(-np.pi, np.pi, (self.num_configs, self.cdf.num_joints))

    def generate_dataset(self):
        print("Generating workspace points...")
        points = self.generate_workspace_points()

        print("Generating configurations...")
        configurations = self.generate_configs()

        print("Computing CDF values for all points and configurations...")
        cdf_values = np.zeros((self.num_configs, len(points)))

        for i, point in enumerate(tqdm(points, desc="Processing points")):
            obstacle = np.concatenate([point, [0]])
            zero_configs, touching_indices = precompute_configs_jax(self.cdf, obstacle, num_precomputed=self.num_zero_configs)
            
            if len(zero_configs) > 0:
                cdf_values[:, i] = compute_cdf_batch(configurations, zero_configs, touching_indices)

        print(f"Dataset generated:")
        print(f"  Configurations shape: {configurations.shape}")
        print(f"  Points shape: {points.shape}")
        print(f"  CDF values shape: {cdf_values.shape}")

        return configurations, points, cdf_values

def save_dataset(configs, points, cdf_values, filename):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the prepared_training_dataset directory relative to the script location
    save_dir = os.path.join(script_dir, 'prepared_training_dataset')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create full path for saving
    full_path = os.path.join(save_dir, filename)
    np.savez(full_path, 
             configurations=configs, 
             points=points, 
             cdf_values=cdf_values)
    print(f"Dataset saved to {full_path}")

def forward_kinematics(q, link_lengths):
    x, y = 0, 0
    positions = [(x, y)]
    angle_sum = 0
    for i, length in enumerate(link_lengths):
        angle_sum += q[i]
        x += length * np.cos(angle_sum)
        y += length * np.sin(angle_sum)
        positions.append((x, y))
    return positions

def visualize_dataset_sample(configs, all_points, cdf_values, sample_idx, save_path, link_lengths):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Get the configuration for this sample
    config = configs[sample_idx]
    
    # Create scatter plot of all points, colored by their CDF values
    cdf_for_config = cdf_values[sample_idx]
    norm = Normalize(vmin=np.min(cdf_for_config), vmax=np.max(cdf_for_config))
    scatter = ax.scatter(all_points[:, 0], all_points[:, 1], c=cdf_for_config, 
                         cmap='viridis', norm=norm, s=20, alpha=0.7)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('CDF Value', rotation=270, labelpad=15)
    
    # Plot robot arm
    joint_positions = forward_kinematics(config, link_lengths)
    for i in range(len(joint_positions) - 1):
        ax.plot([joint_positions[i][0], joint_positions[i+1][0]], 
                [joint_positions[i][1], joint_positions[i+1][1]], 'k-', linewidth=2)
        ax.plot(joint_positions[i][0], joint_positions[i][1], 'ko', markersize=8)
    
    # Plot end effector
    ax.plot(joint_positions[-1][0], joint_positions[-1][1], 'ro', markersize=10)
    
    # Set plot limits based on workspace
    workspace_radius = np.sum(link_lengths)
    ax.set_xlim(-workspace_radius, workspace_radius)
    ax.set_ylim(-workspace_radius, workspace_radius)
    ax.set_title(f'CDF Visualization for {len(link_lengths)}-link Robot Arm (Sample {sample_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    plt.close(fig)  # Close the figure to free up memory

def main():
    # Parameters that can be easily tuned
    link_lengths = [2, 2]  # N-link robot arm
    num_points = 300
    num_configs = 200
    num_zero_configs = 1000 

    print(f"Initializing dataset generation with:")
    print(f"  Number of points: {num_points}")
    print(f"  Number of configurations: {num_configs}")
    print(f"  Number of zero configs per point: {num_zero_configs}")

    generator = CDFDatasetGenerator(link_lengths, num_points, num_configs, num_zero_configs)
    
    start_time = time.time()
    configs, points, cdf_values = generator.generate_dataset()
    end_time = time.time()
    
    print(f"Dataset generation completed in {end_time - start_time:.2f} seconds")
    print(f"Dataset size: {configs.shape[0]}")
    
    print("Saving dataset...")
    save_dataset(configs, points, cdf_values, 'robot_cdf_full_dataset_2_links.npz')

    print("Generating visualizations...")
    for i in tqdm(range(min(2, len(configs))), desc="Generating visualizations"):
        visualize_dataset_sample(configs, points, cdf_values, i, 
                                 f'cdf_dataset/sample_visualization_{i}.png',
                                 link_lengths=link_lengths)

    print("Dataset generation and visualization complete!")

if __name__ == "__main__":
    main()
