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

def point_to_segment_distance(point, segment_start, segment_end):
    """Calculate the minimum distance from a point to a line segment."""
    segment = segment_end - segment_start
    length_sq = np.sum(segment**2)
    if length_sq == 0:
        return np.linalg.norm(point - segment_start)
    
    # Project point onto line
    t = max(0, min(1, np.dot(point - segment_start, segment) / length_sq))
    projection = segment_start + t * segment
    return np.linalg.norm(point - projection)

def is_point_on_robot(point, joint_positions, threshold=0.7):
    """Check if a point is on or near any link of the robot."""
    for i in range(len(joint_positions)-1):
        distance = point_to_segment_distance(point, joint_positions[i], joint_positions[i+1])
        if distance < threshold:
            return True
    return False

def generate_zero_config_dataset(link_lengths, num_configs, points_per_link, points_per_link_nearby, 
                               off_robot_points_per_config=50, nearby_samples_per_joint=5):
    configs = []
    points = []
    cdf_values = []  # Will now store scalar values instead of vectors
    
    workspace_radius = sum(link_lengths) * 1.2
    
    for _ in range(num_configs):
        # Original configuration (CDF = 0)
        q = np.random.uniform(-np.pi, np.pi, len(link_lengths))
        
        # Sample points for original config
        joint_positions = forward_kinematics(q, link_lengths)
        for i in range(len(link_lengths)):
            start = joint_positions[i]
            end = joint_positions[i+1]
            t_values = np.linspace(0, 1, points_per_link)
            for t in t_values:
                point = sample_point_on_edge(start, end, t)
                configs.append(q.copy())
                points.append(point)
                cdf_values.append(0.0)  # Scalar value for zero config
        
        # Generate off-robot points (CDF = 100)
        off_robot_count = 0
        max_attempts = off_robot_points_per_config * 10
        attempts = 0
        
        while off_robot_count < off_robot_points_per_config and attempts < max_attempts:
            theta = np.random.uniform(0, 2*np.pi)
            r = np.sqrt(np.random.uniform(0, 1)) * workspace_radius
            point = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            if not is_point_on_robot(point, joint_positions):
                configs.append(q.copy())
                points.append(point)
                cdf_values.append(100.0)  # Scalar value for off-robot points
                off_robot_count += 1
            
            attempts += 1
        
        # Sample points for nearby configs (truncated CDF values)
        for joint_idx in range(len(link_lengths)):
            deltas = np.random.uniform(-0.3, 0.3, nearby_samples_per_joint)
            
            for delta in deltas:
                nearby_q = q.copy()
                nearby_q[joint_idx] += delta
                
                # Get positions for both original and perturbed configurations
                original_positions = forward_kinematics(q, link_lengths)
                perturbed_positions = forward_kinematics(nearby_q, link_lengths)
                
                # Sample points from this joint's link and all subsequent links
                for link_idx in range(joint_idx, len(link_lengths)):
                    start = perturbed_positions[link_idx]
                    end = perturbed_positions[link_idx+1]
                    t_values = np.linspace(0, 1, points_per_link_nearby)
                    for t in t_values:
                        point = sample_point_on_edge(start, end, t)
                        
                        # Calculate distances to all links in both configurations
                        original_distances = []
                        perturbed_distances = []
                        for i in range(len(link_lengths)):
                            # Distance to links in original configuration
                            dist_original = point_to_segment_distance(
                                point, 
                                original_positions[i], 
                                original_positions[i+1]
                            )
                            original_distances.append(dist_original)
                            
                            # Distance to links in perturbed configuration
                            dist_perturbed = point_to_segment_distance(
                                point, 
                                perturbed_positions[i], 
                                perturbed_positions[i+1]
                            )
                            perturbed_distances.append(dist_perturbed)
                        
                        # Check if point is closest to the intended link in both configurations
                        closest_link_original = np.argmin(original_distances)
                        closest_link_perturbed = np.argmin(perturbed_distances)
                        
                        # Only add point if it's closest to the intended link in both configurations
                        if closest_link_original == link_idx and closest_link_perturbed == link_idx:
                            configs.append(q.copy())
                            points.append(point)
                            cdf_values.append(abs(delta))
    
    return np.array(configs), np.array(points), np.array(cdf_values)

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

def visualize_samples(configs, points, cdf_values, link_lengths, num_configs_to_plot=5):
    plt.figure(figsize=(15, 3*num_configs_to_plot))
    
    # Get indices of unique configurations
    _, unique_indices = np.unique(configs, axis=0, return_index=True)
    selected_indices = np.random.choice(unique_indices, num_configs_to_plot, replace=False)
    
    for idx, config_idx in enumerate(selected_indices):
        plt.subplot(num_configs_to_plot, 1, idx+1)
        
        # Get the configuration
        config = configs[config_idx]
        
        # Get all points for this config
        config_mask = np.all(configs == config, axis=1)
        config_points = points[config_mask]
        config_cdfs = cdf_values[config_mask]
        
        # Separate different types of points
        zero_cdf_mask = config_cdfs == 0  # Points on robot
        one_cdf_mask = config_cdfs == 100  # Off-robot points
        truncated_mask = ~(zero_cdf_mask | one_cdf_mask)  # Points from nearby configs
        
        print(f"\nConfig {idx}: {config}")
        print(f"Total points for this config: {len(config_points)}")
        print(f"On-robot points: {np.sum(zero_cdf_mask)}")
        print(f"Off-robot points: {np.sum(one_cdf_mask)}")
        print(f"Truncated points: {np.sum(truncated_mask)}")
        
        # Plot points
        plt.scatter(config_points[zero_cdf_mask][:, 0], 
                   config_points[zero_cdf_mask][:, 1], 
                   c='blue', alpha=0.5, label='On Robot (CDF=0)')
        plt.scatter(config_points[one_cdf_mask][:, 0], 
                   config_points[one_cdf_mask][:, 1], 
                   c='red', alpha=0.5, label='Off Robot')
        if np.any(truncated_mask):
            plt.scatter(config_points[truncated_mask][:, 0], 
                       config_points[truncated_mask][:, 1], 
                       c='green', alpha=0.5, label='Truncated')
        
        # Plot robot arm
        joint_positions = forward_kinematics(config, link_lengths)
        plt.plot(joint_positions[:, 0], joint_positions[:, 1], 'k-', linewidth=2)
        plt.plot(joint_positions[:, 0], joint_positions[:, 1], 'ko')
        
        plt.title(f'Configuration: [{config}]')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('cdf_dataset/visualization.png')
    plt.close()

def main():
    link_lengths = [2, 2]  # 1-link robot arm
    num_configs = 200
    points_per_link = 20      # Points for original configs
    nearby_samples_per_joint = 10

    points_per_link_nearby = 10

    off_robot_points_per_config = 100  # New parameter
    
    print(f"Generating dataset with:")
    print(f"  Number of configurations: {num_configs}")
    print(f"  Points per link (original): {points_per_link}")
    print(f"  Points per link (nearby): {points_per_link_nearby}")
    print(f"  Nearby samples per joint: {nearby_samples_per_joint}")
    print(f"  Total points per original config: {points_per_link}")
    print(f"  Total points per nearby config: {points_per_link_nearby}")
    print(f"  Total points from nearby configs: {points_per_link_nearby * nearby_samples_per_joint}")
    
    configs, points, cdf_values = generate_zero_config_dataset(
        link_lengths, num_configs, points_per_link, points_per_link_nearby, 
        off_robot_points_per_config=off_robot_points_per_config, 
        nearby_samples_per_joint=nearby_samples_per_joint)
    
    print("Saving dataset...")
    save_dataset(configs, points, cdf_values, 'robot_cdf_truncated_dataset_new.npz')
    
    print("Generating visualizations...")
    visualize_samples(configs, points, cdf_values, link_lengths)
    
    print("Dataset generation and visualization complete!")

if __name__ == "__main__":
    main()
