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

def calculate_cdf(point, theta):
    """Calculate CDF value and rotation direction.
    Returns:
        tuple: (cdf_value, rotation_sign)
        rotation_sign is +1 for counterclockwise, -1 for clockwise
    """
    x, y = point
    if x == 0 and y == 0:
        return np.pi, 1  # Default direction for origin point
    
    point_angle = np.arctan2(y, x)
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    # Calculate signed difference
    diff = point_angle - theta
    # Normalize to [-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    
    # Return absolute difference and sign
    return abs(diff), np.sign(diff)

def get_link2_points_with_cdf(joint_positions, theta1, theta2, cdf_value, link_lengths, num_points=3):
    """Generate points on link 2 after rotating link 1 by cdf_value.
    
    Args:
        joint_positions: Original joint positions
        theta1: Original angle of link 1
        theta2: Angle of link 2 relative to link 1
        cdf_value: The CDF value computed for a point
        link_lengths: List of link lengths
        num_points: Number of points to sample along link 2
    """
    # Rotate link 1 by cdf_value
    new_theta1 = theta1 + cdf_value
    # Calculate new joint positions
    new_joint_pos = forward_kinematics(np.array([new_theta1, theta2]), link_lengths)
    
    # Sample points along the new link 2 position
    start = new_joint_pos[1]  # Second joint position
    end = new_joint_pos[2]    # End effector position
    t_values = np.linspace(0, 1, num_points)
    
    points = []
    for t in t_values:
        point = sample_point_on_edge(start, end, t)
        points.append(point)
        
    return points

def is_within_link1_workspace(point, link1_radius):
    """Check if point is within link 1's workspace."""
    return np.linalg.norm(point) <= link1_radius

def generate_dataset_2_link(link_lengths, num_configs=200, points_per_link=50, points_with_cdf=300,
                          off_robot_points=50, extra_points_per_cdf=3,
                          nearby_samples_per_joint=10, points_per_link_nearby=10):
    """Generate dataset for 2-link robot with CDF values for both links.
    
    Args:
        link_lengths: List of two link lengths
        num_configs: Number of random configurations
        points_per_link: Number of points per link
        points_with_cdf: Number of points for CDF calculation per link
        off_robot_points: Number of off-robot points per config
    """
    configs = []
    points = []
    cdf_values = []
    
    link1_workspace_radius = link_lengths[0] * 1.0
    link2_workspace_radius = link_lengths[1] * 1.0
    workspace_radius = sum(link_lengths) * 1.0
    
    for _ in tqdm(range(num_configs)):
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        q = np.array([theta1, theta2])
        joint_positions = forward_kinematics(q, link_lengths)
        
        # 1. Points on both links (zero CDF)
        for link_idx in range(2):
            start = joint_positions[link_idx]
            end = joint_positions[link_idx + 1]
            t_values = np.linspace(0, 1, points_per_link)
            
            for t in t_values:
                point = sample_point_on_edge(start, end, t)
                configs.append(q)
                points.append(point)
                cdf_values.append(0.0)
        
        # Points with CDF values (only for link 1)
        points_added = 0
        max_attempts = points_with_cdf * 10
        attempts = 0
        
        while points_added < points_with_cdf and attempts < max_attempts:
            r = np.sqrt(np.random.uniform(0, 1)) * link1_workspace_radius
            phi = np.random.uniform(0, 2*np.pi)
            point = np.array([r * np.cos(phi), r * np.sin(phi)])
            
            # Calculate distances to both links
            dist_to_link1 = point_to_segment_distance(point, joint_positions[0], joint_positions[1])
            dist_to_link2 = point_to_segment_distance(point, joint_positions[1], joint_positions[2])
            
            if dist_to_link1 < dist_to_link2:
                # Calculate CDF and rotation direction for the point
                cdf_value, rotation_sign = calculate_cdf(point, theta1)
                
                # Add original point
                configs.append(q)
                points.append(point)
                cdf_values.append(cdf_value)
                
                # Generate additional points on link 2 with same CDF
                extra_points = get_link2_points_with_cdf(
                    joint_positions, theta1, theta2, 
                    rotation_sign * cdf_value,
                    link_lengths, extra_points_per_cdf)
                
                # Process each extra point
                for extra_point in extra_points:
                    if is_within_link1_workspace(extra_point, link1_workspace_radius):
                        # If within link1's workspace, compute direct CDF and take minimum
                        direct_cdf, _ = calculate_cdf(extra_point, theta1)
                        point_cdf = min(cdf_value, direct_cdf)
                    else:
                        # If outside link1's workspace, use original CDF value
                        point_cdf = cdf_value
                    
                    configs.append(q)
                    points.append(extra_point)
                    cdf_values.append(point_cdf)
                
                points_added += 1
            attempts += 1
        

        
        # 3. Off-robot points (CDF = 100)
        off_robot_count = 0
        max_attempts = off_robot_points * 10
        attempts = 0
        
        while off_robot_count < off_robot_points and attempts < max_attempts:
            r = np.sqrt(np.random.uniform(0, 1)) * workspace_radius
            phi = np.random.uniform(0, 2*np.pi)
            point = np.array([r * np.cos(phi), r * np.sin(phi)])
            
            # Check if point is far from both links
            dist_to_link1 = point_to_segment_distance(point, joint_positions[0], joint_positions[1])
            dist_to_link2 = point_to_segment_distance(point, joint_positions[1], joint_positions[2])
            
            if dist_to_link1 > 0.3 and dist_to_link2 > 0.3:  # Threshold distance from links
                configs.append(q)
                points.append(point)
                cdf_values.append(100.0)  # Fixed value for off-robot points
                off_robot_count += 1
            attempts += 1
        
        # Add perturbed configurations for link 2
        target_truncated_points = nearby_samples_per_joint * points_per_link_nearby  # Total desired points per configuration
        num_perturbations = nearby_samples_per_joint * 5       # Generate more perturbations than needed
        points_per_perturbation = points_per_link_nearby  # Points per perturbation
        
        candidate_points = []
        candidate_cdfs = []
        
        # Generate candidate points with perturbations
        deltas = np.random.uniform(-0.15, 0.15, num_perturbations)
        
        for delta in deltas:
            nearby_q = q.copy()
            nearby_q[1] += delta  # Only perturb link 2
            
            nearby_positions = forward_kinematics(nearby_q, link_lengths)
            original_positions = forward_kinematics(q, link_lengths)
            
            # Sample points along perturbed link 2
            t_values = np.linspace(0, 1, points_per_perturbation)
            for t in t_values:
                point = sample_point_on_edge(nearby_positions[1], nearby_positions[2], t)
                
                # Check if point is closer to link 2
                dist_to_link1 = point_to_segment_distance(
                    point, 
                    original_positions[0], 
                    original_positions[1]
                )
                dist_to_link2 = point_to_segment_distance(
                    point, 
                    original_positions[1], 
                    original_positions[2]
                )
                
                if dist_to_link2 < dist_to_link1:
                    candidate_points.append(point)
                    candidate_cdfs.append(abs(delta))
        
        # Add the valid points up to our target number
        num_to_add = min(len(candidate_points), target_truncated_points)
        if num_to_add > 0:
            # Randomly select points if we have more than needed
            indices = np.random.choice(len(candidate_points), num_to_add, replace=False)
            for idx in indices:
                configs.append(q.copy())
                points.append(candidate_points[idx])
                cdf_values.append(candidate_cdfs[idx])
        else:
            print(f"Warning: Could not generate any valid truncated points for configuration {_}")
    
    return np.array(configs), np.array(points), np.array(cdf_values)


def save_dataset(configs, points, cdf_values, filename):
    """Save dataset to file."""
    os.makedirs('cdf_dataset', exist_ok=True)
    full_path = os.path.join('cdf_dataset', filename)
    np.savez(full_path, 
             configurations=configs, 
             points=points, 
             cdf_values=cdf_values)
    print(f"Dataset saved to {full_path}")

def visualize_samples(configs, points, cdf_values, link_length, num_configs_to_plot=5):
    """Visualize sample configurations and their points."""
    plt.figure(figsize=(15, 3*num_configs_to_plot))
    
    # Get indices of unique configurations
    unique_configs = np.unique(configs, axis=0)
    selected_configs = unique_configs[np.random.choice(len(unique_configs), num_configs_to_plot, replace=False)]
    
    for idx, theta in enumerate(selected_configs):
        plt.subplot(num_configs_to_plot, 1, idx+1)
        
        # Find all points for this configuration
        config_mask = np.all(configs == theta, axis=1)
        config_points = points[config_mask]
        config_cdfs = cdf_values[config_mask]
        
        # Plot points colored by CDF value
        scatter = plt.scatter(config_points[:, 0], config_points[:, 1], 
                            c=config_cdfs, cmap='viridis', 
                            alpha=0.6, label='Points')
        plt.colorbar(scatter, label='CDF Value')
        
        # Plot robot links
        joint_positions = forward_kinematics(theta, link_length)
        plt.plot(joint_positions[:, 0], joint_positions[:, 1], 'k-', linewidth=2)
        plt.plot(joint_positions[:, 0], joint_positions[:, 1], 'ko')
        
        plt.title(f'Configuration: theta1={theta[0]:.2f}, theta2={theta[1]:.2f}')
        plt.axis('equal')
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('cdf_dataset/visualization.png')
    plt.close()

def main():
    # Parameters
    link_length = [2.0, 2.0]
    num_configs = 200
    points_per_link = 30
    points_with_cdf = 50
    off_robot_points = 100
    extra_points_per_cdf = 0
    nearby_samples_per_joint = 10  # New parameter
    points_per_link_nearby = 10    # New parameter
    
    print("Generating dataset...")
    configs, points, cdf_values = generate_dataset_2_link(
        link_length, 
        num_configs=num_configs, 
        points_per_link=points_per_link, 
        points_with_cdf=points_with_cdf,
        off_robot_points=off_robot_points,
        extra_points_per_cdf=extra_points_per_cdf,
        nearby_samples_per_joint=nearby_samples_per_joint,
        points_per_link_nearby=points_per_link_nearby
    )
    
    print("Saving dataset...")
    save_dataset(configs, points, cdf_values, 'robot_cdf_incremental_truncated_dataset_2_links.npz')
    
    print("Generating visualizations...")
    visualize_samples(configs, points, cdf_values, link_length)
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main() 