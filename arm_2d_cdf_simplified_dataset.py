import numpy as np
from scipy.optimize import minimize
import time
from arm_2d_cdf_simplified import SimpleCDF2D
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class CDFDatasetGenerator:
    def __init__(self, link_lengths, num_points, num_configs, num_zero_configs):
        self.cdf = SimpleCDF2D(link_lengths, num_precomputed=num_zero_configs)
        self.num_points = num_points
        self.num_configs = num_configs
        self.workspace_radius = np.sum(link_lengths)

    def generate_workspace_points(self):
        angles = np.random.uniform(0, 2*np.pi, self.num_points)
        radii = np.random.uniform(0, self.workspace_radius, self.num_points)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def generate_configs(self):
        return np.random.uniform(-np.pi, np.pi, (self.num_configs, self.cdf.num_joints))

    def compute_zero_configs(self, point):
        obstacle = [point[0], point[1], 0]
        self.cdf.precompute_configs(obstacle)
        return self.cdf.precomputed_configs

    def compute_cdf(self, config, zero_configs):
        distances = np.linalg.norm(config - zero_configs, axis=1)
        return np.min(distances)

    def generate_dataset(self):
        print("Generating workspace points...")
        points = self.generate_workspace_points()

        print("Generating configurations...")
        configs = self.generate_configs()

        print("Computing zero configurations for each point...")
        zero_configs_dict = {}
        valid_points = []
        for point in tqdm(points, desc="Computing zero configs"):
            zero_configs = self.compute_zero_configs(point)
            if len(zero_configs) > 0:
                zero_configs_dict[tuple(point)] = zero_configs
                valid_points.append(point)

        valid_points = np.array(valid_points)

        print("Generating dataset...")
        cdf_values = np.zeros((self.num_configs, len(valid_points)))
        total_pairs = self.num_configs * len(valid_points)
        with tqdm(total=total_pairs, desc="Computing CDF values") as pbar:
            for i, config in enumerate(configs):
                for j, point in enumerate(valid_points):
                    cdf_values[i, j] = self.compute_cdf(config, zero_configs_dict[tuple(point)])
                    pbar.update(1)

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

def visualize_dataset_sample(configs, all_points, cdf_values, sample_idx, save_path):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Get the configuration and all points for this sample
    config = configs[sample_idx]
    points = all_points
    
    # Create scatter plot
    scatter = ax.scatter(points[:, 0], points[:, 1], c=cdf_values[sample_idx], cmap='viridis', s=10)
    
    # Plot robot arm
    joint_positions = forward_kinematics(config)
    ax.plot([0, joint_positions[0][0]], [0, joint_positions[0][1]], 'k-', linewidth=2)
    ax.plot([joint_positions[0][0], joint_positions[1][0]], [joint_positions[0][1], joint_positions[1][1]], 'k-', linewidth=2)
    ax.plot(0, 0, 'ko', markersize=8)
    ax.plot(joint_positions[0][0], joint_positions[0][1], 'ko', markersize=8)
    ax.plot(joint_positions[1][0], joint_positions[1][1], 'ko', markersize=8)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title(f'CDF Visualization for Robot Arm (Sample {sample_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('CDF Value')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")

def forward_kinematics(q):
    l1, l2 = 2, 2  # link lengths
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])
    return [(x1, y1), (x2, y2)]

def main():
    # Parameters that can be easily tuned
    link_lengths = [2, 2]  # 2-link robot arm
    num_points = 2000
    num_configs = 200
    num_zero_configs = 100  # Number of zero configs to compute for each point

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
    save_dataset(configs, all_points, cdf_values, 'cdf_dataset/robot_cdf_dataset_2_links.npy')

    print("Generating visualizations...")
    for i in tqdm(range(5), desc="Generating visualizations"):
        visualize_dataset_sample(configs, all_points, cdf_values, i, f'cdf_dataset/sample_visualization_{i}.png')

    print("Dataset generation and visualization complete!")

if __name__ == "__main__":
    main()
