import numpy as np
import torch
from pathlib import Path

def farthest_point_sampling(points, n_samples):
    """
    Perform farthest point sampling on configurations
    Args:
        points: [N, D] tensor of configurations
        n_samples: number of points to sample
    Returns:
        [n_samples] indices of sampled points
    """
    points = torch.tensor(points)
    N = points.shape[0]
    selected_indices = torch.zeros(n_samples, dtype=torch.long)
    
    # Initialize with a random point
    selected_indices[0] = torch.randint(N, (1,))
    
    # Compute distances to the first point
    distances = torch.norm(points - points[selected_indices[0]], dim=1)
    
    # Iteratively select farthest points
    for i in range(1, n_samples):
        # Select the point with maximum distance
        new_idx = torch.argmax(distances)
        selected_indices[i] = new_idx
        
        # Update distances
        new_distances = torch.norm(points - points[new_idx], dim=1)
        distances = torch.minimum(distances, new_distances)
    
    return selected_indices.numpy()

def refine_contact_database(input_path, output_path, min_configs=100, max_configs=500):
    """
    Refine contact database by filtering points and downsampling configurations
    Args:
        input_path: Path to input contact database
        output_path: Path to save refined database
        min_configs: Minimum number of configurations required per point
        max_configs: Maximum number of configurations to keep per point
    """
    # Load original database
    print(f"Loading database from {input_path}")
    db = np.load(input_path, allow_pickle=True).item()
    
    # Get original statistics
    n_points_original = len(db['points'])
    configs_per_point = [len(configs) for configs in db['contact_configs']]
    print(f"\nOriginal database statistics:")
    print(f"Total points: {n_points_original}")
    print(f"Configurations per point: min={min(configs_per_point)}, "
          f"max={max(configs_per_point)}, "
          f"mean={np.mean(configs_per_point):.1f}")
    
    # Filter points with sufficient configurations
    valid_indices = [i for i, configs in enumerate(db['contact_configs']) 
                    if len(configs) >= min_configs]
    
    # Create new database
    refined_db = {
        'points': db['points'][valid_indices],
        'contact_configs': [],
        'link_indices': []
    }
    
    # Process each valid point
    print("\nProcessing points...")
    for idx in valid_indices:
        configs = db['contact_configs'][idx]
        link_idx = db['link_indices'][idx]
        
        if len(configs) > max_configs:
            # Perform FPS sampling
            sampled_indices = farthest_point_sampling(configs, max_configs)
            configs = configs[sampled_indices]
            link_idx = link_idx[sampled_indices]
        
        refined_db['contact_configs'].append(configs)
        refined_db['link_indices'].append(link_idx)
    
    # Get refined statistics
    n_points_refined = len(refined_db['points'])
    configs_per_point_refined = [len(configs) for configs in refined_db['contact_configs']]
    print(f"\nRefined database statistics:")
    print(f"Total points: {n_points_refined}")
    print(f"Configurations per point: min={min(configs_per_point_refined)}, "
          f"max={max(configs_per_point_refined)}, "
          f"mean={np.mean(configs_per_point_refined):.1f}")
    
    # Save refined database
    print(f"\nSaving refined database to {output_path}")
    np.save(output_path, refined_db)
    print("Done!")

if __name__ == "__main__":
    src_dir = Path(__file__).parent
    input_path = src_dir / "data" / "contact_db_2d.npy"
    output_path = src_dir / "data" / "contact_db_2d_refined.npy"
    
    refine_contact_database(
        input_path=input_path,
        output_path=output_path,
        min_configs=100,
        max_configs=500
    ) 