import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
    configs = []
    points = []
    
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
                
                configs.append(q)
                points.append(point)
    
    return np.array(configs), np.array(points)

def save_dataset(configs, points, filename):
    os.makedirs('cdf_dataset', exist_ok=True)
    full_path = os.path.join('cdf_dataset', filename)
    data = {
        'configurations': configs,
        'points': points,
        'cdf_values': np.zeros(len(configs))  # All distances are 0
    }
    np.save(full_path, data)
    print(f"Dataset saved to {full_path}")

def visualize_samples(configs, points, link_lengths, num_samples=1):
    for i in range(num_samples):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Sample a random configuration
        config_idx = np.random.randint(0, len(configs))
        config = configs[config_idx]
        
        # Plot the robot arm
        joint_positions = forward_kinematics(config, link_lengths)
        for j in range(len(joint_positions) - 1):
            ax.plot([joint_positions[j][0], joint_positions[j+1][0]], 
                    [joint_positions[j][1], joint_positions[j+1][1]], 'k-', linewidth=2)
            ax.plot(joint_positions[j][0], joint_positions[j][1], 'ko', markersize=5)
        
        # Plot all points corresponding to this configuration
        mask = np.all(configs == config, axis=1)
        config_points = points[mask]
        ax.scatter(config_points[:, 0], config_points[:, 1], c='red', s=20, alpha=0.5)
        
        ax.set_aspect('equal')
        ax.set_title(f'Sample {i+1}: Robot Configuration and Zero-Distance Points')
        plt.savefig(f'cdf_dataset/zero_config_visualization_{i+1}.png')
        plt.close()

def main():
    link_lengths = [2, 2]  # 4-link robot arm
    num_configs = 1000
    points_per_config = 200
    
    print(f"Generating dataset with:")
    print(f"  Number of configurations: {num_configs}")
    print(f"  Points per configuration: {points_per_config}")
    print(f"  Total data points: {num_configs * points_per_config}")
    
    configs, points = generate_zero_config_dataset(link_lengths, num_configs, points_per_config)
    
    print("Saving dataset...")
    save_dataset(configs, points, 'robot_cdf_zeroconfigs_dataset_2_links.npy')
    
    print("Generating visualizations...")
    visualize_samples(configs, points, link_lengths)
    
    print("Dataset generation and visualization complete!")

if __name__ == "__main__":
    main()
