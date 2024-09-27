import numpy as np
import matplotlib.pyplot as plt
from arm_2d_utils import forward_kinematics, transform_shape
from data.arm_2d_config import shapes, NUM_LINKS

def load_dataset(file_path):
    """Load the dataset from the specified file."""
    return np.load(file_path, allow_pickle=True).item()

def visualize_dataset_sample(configs, all_points, cdf_values, sample_idx, save_path):
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
    # Load the dataset
    dataset_path = 'cdf_dataset/robot_cdf_dataset_2_links.npy'
    data = load_dataset(dataset_path)
    
    configs = data['configurations']
    all_points = data['points']
    cdf_values = data['cdf_values']
    
    # Visualize multiple samples
    num_samples = 5
    for i in range(num_samples):
        sample_idx = np.random.randint(0, len(configs))
        save_path = f'cdf_dataset/dataset_visualization_sample_{i}.png'
        visualize_dataset_sample(configs, all_points, cdf_values, sample_idx, save_path)

if __name__ == "__main__":
    main()