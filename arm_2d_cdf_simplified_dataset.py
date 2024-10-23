import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time
from arm_2d_cdf_simplified import SimpleCDF2D, precompute_configs_jax
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

@jax.jit
def compute_cdf_batch(configs, zero_configs):
    return jax.vmap(lambda c: jnp.min(jnp.linalg.norm(c - zero_configs, axis=1)))(configs)

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
        configs = self.generate_configs()

        print("Computing zero configurations and CDF values for all points...")
        valid_points = []
        cdf_values = []

        for point in tqdm(points, desc="Processing points"):
            obstacle = np.concatenate([point, [0]])
            zero_configs = precompute_configs_jax(self.cdf, obstacle, num_precomputed=self.num_zero_configs)
            
            if len(zero_configs) > 0:
                point_cdf_values = compute_cdf_batch(configs, zero_configs)
                cdf_values.append(point_cdf_values)
                valid_points.append(point)

        valid_points = np.array(valid_points)
        cdf_values = np.array(cdf_values).T  # Transpose to match the expected shape

        return configs, valid_points, cdf_values

def save_dataset(configs, all_points, cdf_values, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = {
        'configurations': configs,
        'points': all_points,
        'cdf_values': cdf_values
    }
    np.save(filename, data)
    print(f"Dataset saved to {filename}")

def forward_kinematics(q, link_lengths):
    x, y = 0, 0
    positions = [(x, y)]
    for i in range(len(link_lengths)):
        x += link_lengths[i] * np.cos(np.sum(q[:i+1]))
        y += link_lengths[i] * np.sin(np.sum(q[:i+1]))
        positions.append((x, y))
    return positions

def visualize_dataset_sample(configs, all_points, cdf_values, sample_idx, save_path, link_lengths):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Get the point and zero configuration for this sample
    point = all_points[sample_idx]
    config = configs[sample_idx][0]  # Take the first zero config for visualization
    
    # Create scatter plot of all points
    ax.scatter(all_points[:, 0], all_points[:, 1], c='lightblue', s=10, alpha=0.5)
    
    # Highlight the selected point
    ax.scatter(point[0], point[1], c='red', s=50, zorder=3)
    
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
    ax.set_title(f'Zero Configuration Visualization for {len(link_lengths)}-link Robot Arm (Sample {sample_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")



def main():
    # Parameters that can be easily tuned
    link_lengths = [2, 2]  # N-link robot arm
    num_points = 20
    num_configs = 100
    num_zero_configs = 20 # Number of zero configs to compute for each point

    print(f"Initializing dataset generation with:")
    print(f"  Number of points: {num_points}")
    print(f"  Number of configurations: {num_configs}")
    print(f"  Number of zero configs per point: {num_zero_configs}")

    generator = CDFDatasetGenerator(link_lengths, num_points, num_configs, num_zero_configs)
    
    start_time = time.time()
    configs, all_points, cdf_values = generator.generate_dataset()
    end_time = time.time()
    
    print(f"Dataset generation completed in {end_time - start_time:.2f} seconds")
    print(f"Dataset size: {configs.shape[0]} configurations, {all_points.shape[0]} points")
    
    print("Saving dataset...")
    save_dataset(configs, all_points, cdf_values, 'cdf_dataset/robot_cdf_dataset_4_links.npy')

    print("Generating visualizations...")
    for i in tqdm(range(5), desc="Generating visualizations"):
        visualize_dataset_sample(configs, all_points, cdf_values, i, 
                                 f'cdf_dataset/sample_visualization_{i}.png',
                                 link_lengths=link_lengths)

    print("Dataset generation and visualization complete!")

if __name__ == "__main__":
    main()
