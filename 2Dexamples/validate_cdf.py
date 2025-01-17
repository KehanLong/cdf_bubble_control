import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from arm_2d_cdf_simplified import SimpleCDF2D, precompute_configs_jax
from tqdm import tqdm

from arm_2d_cdf_simplified_dataset import compute_cdf_batch



def forward_kinematics(q, link_lengths):
    """Compute forward kinematics for a planar robot arm."""
    x, y = 0, 0
    positions = [(x, y)]
    angle_sum = 0
    for i, length in enumerate(link_lengths):
        angle_sum += q[i]
        x += length * np.cos(angle_sum)
        y += length * np.sin(angle_sum)
        positions.append((x, y))
    return positions

def visualize_worst_cases(worst_cases, link_lengths):
    """Visualize the worst cases with robot configurations and points."""
    plt.figure(figsize=(15, 3*len(worst_cases)))
    
    for idx, case in enumerate(worst_cases):
        plt.subplot(len(worst_cases), 1, idx+1)
        
        # Get configuration and point
        config = case['config']
        point = case['point']
        
        # Plot robot configuration
        joint_positions = forward_kinematics(config, link_lengths)
        plt.plot([joint_positions[i][0] for i in range(len(joint_positions))],
                [joint_positions[i][1] for i in range(len(joint_positions))], 
                'k-', linewidth=2, label='Robot Links')
        plt.plot([joint_positions[i][0] for i in range(len(joint_positions))],
                [joint_positions[i][1] for i in range(len(joint_positions))], 
                'ko', markersize=8)
        
        # Plot point
        plt.plot(point[0], point[1], 'r*', markersize=15, label='Point')
        
        plt.title(f'Case {idx+1}: Config=[{config[0]:.2f}, {config[1]:.2f}], '
                 f'Geometric CDF={case["geometric_cdf"]:.2f}, BFGS CDF={case["bfgs_cdf"]:.2f}')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('cdf_validation_worst_cases.png')
    plt.close()

def validate_and_refine_dataset(link_lengths, configs, points, geometric_cdfs, diff_threshold=0.05):
    """Validate CDF values and filter out pairs with large differences."""
    # Initialize BFGS calculator
    cdf_calculator = SimpleCDF2D(link_lengths)
    num_zero_configs = 100
    
    refined_indices = []
    differences = []
    print("Validating and refining dataset...")
    
    for idx in tqdm(range(len(points))):
        point = points[idx]
        geometric_cdf = geometric_cdfs[idx]
        config = configs[idx]
        
        # Keep points with CDF = 0 (on robot) or 100 (off robot)
        if geometric_cdf in [0, 100]:
            refined_indices.append(idx)
            continue
            
        # Compute BFGS-based CDF
        obstacle = np.concatenate([point, [0]])
        zero_configs, touching_indices = precompute_configs_jax(
            cdf_calculator, 
            obstacle, 
            num_precomputed=num_zero_configs
        )
        
        if len(zero_configs) > 0:
            # Compute minimum distance to zero configs
            distances = []
            for zero_config, touch_idx in zip(zero_configs, touching_indices):
                mask = np.arange(config.shape[0]) <= touch_idx
                diff = (config - zero_config) * mask
                distances.append(np.linalg.norm(diff))
            
            bfgs_cdf = min(distances)
            difference = abs(geometric_cdf - bfgs_cdf)
            
            # Store difference and index if below threshold
            if difference <= diff_threshold:
                refined_indices.append(idx)
            
            differences.append({
                'point': point,
                'config': config,
                'geometric_cdf': geometric_cdf,
                'bfgs_cdf': bfgs_cdf,
                'difference': difference
            })
    
    # Convert to numpy array for indexing
    refined_indices = np.array(refined_indices)
    
    # Create refined dataset
    refined_configs = configs[refined_indices]
    refined_points = points[refined_indices]
    refined_cdfs = geometric_cdfs[refined_indices]
    
    print("\nRefinement Results:")
    print(f"Original dataset size: {len(configs)}")
    print(f"Refined dataset size: {len(refined_configs)}")
    print(f"Removed {len(configs) - len(refined_configs)} pairs ({((len(configs) - len(refined_configs))/len(configs))*100:.2f}%)")
    
    # Save refined dataset
    np.savez('cdf_dataset/robot_cdf_incremental_refined_dataset_2_links.npz',
             configurations=refined_configs,
             points=refined_points,
             cdf_values=refined_cdfs)
    
    return differences

def main():
    # Load the original dataset
    print("Loading dataset...")
    data = np.load('cdf_dataset/robot_cdf_incremental_dataset_2_links.npz')
    loaded_configurations = data['configurations']
    loaded_points = data['points']
    loaded_cdf_values = data['cdf_values']
    
    link_lengths = [2.0, 2.0]
    
    print("\nValidating and refining dataset...")
    differences = validate_and_refine_dataset(
        link_lengths=link_lengths,
        configs=loaded_configurations,
        points=loaded_points,
        geometric_cdfs=loaded_cdf_values,
        diff_threshold=0.05
    )
    
    print("Dataset refinement complete!")

if __name__ == "__main__":
    main()