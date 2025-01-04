import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.xarm_model import XArmFK
from robot_sdf import RobotSDF
from typing import Optional, Tuple

def find_goal_configuration(
    goal_pos: torch.Tensor,
    points_robot: torch.Tensor,
    robot_fk: Optional[XArmFK] = None,
    robot_sdf: Optional[RobotSDF] = None,
    n_samples: int = 100000,
    threshold: float = 0.1,
    device: str = 'cuda'
) -> Tuple[Optional[np.ndarray], float]:
    """Find a valid goal configuration for a given goal position using random sampling.
    
    Args:
        goal_pos: Target position tensor [3]
        points_robot: Point cloud of robot environment [N, 3]
        robot_fk: XArm forward kinematics model (will create new one if None)
        robot_sdf: Robot SDF model (will create new one if None)
        n_samples: Number of random samples to try
        threshold: Maximum acceptable distance to goal position
        device: Device to use for computations ('cuda' or 'cpu')
    
    Returns:
        tuple: (goal_configuration, achieved_distance) or (None, float('inf')) if no solution found
        - goal_configuration: Joint angles achieving the goal position [6] or None
        - achieved_distance: Distance to goal position or inf if no solution found
    """
    # Initialize models if not provided
    if robot_fk is None:
        robot_fk = XArmFK(device=device)
    if robot_sdf is None:
        robot_sdf = RobotSDF(device=device)

    print(f"\nSearching for goal configuration:")
    print(f"Target position: {goal_pos.cpu().numpy()}")
    
    # Get joint limits and ensure float32
    lower_limits = robot_fk.joint_limits[:, 0].cpu().numpy().astype(np.float32)
    upper_limits = robot_fk.joint_limits[:, 1].cpu().numpy().astype(np.float32)
    
    # Sample configurations in batches for efficiency
    batch_size = 1000
    n_batches = int(n_samples // batch_size)
    
    best_config = None
    best_dist = float('inf')
    
    # Ensure point cloud is properly formatted
    points_robot = points_robot.to(dtype=torch.float32, device=device)
    while points_robot.dim() > 2:
        points_robot = points_robot.squeeze(0)
    
    for batch in range(n_batches):
        # Sample random configurations
        configs = np.random.uniform(
            low=lower_limits,
            high=upper_limits,
            size=(batch_size, len(lower_limits))
        ).astype(np.float32)
        
        configs_tensor = torch.tensor(configs, device=device, dtype=torch.float32)
        
        try:
            # Get end-effector positions
            ee_positions = robot_fk.fkine(configs_tensor)[:, -1]  # [batch_size, 3]
            
            # Compute distances to goal
            distances = torch.norm(ee_positions - goal_pos.unsqueeze(0), dim=1)
            
            # Find valid configurations
            valid_indices = torch.where(distances < threshold)[0]
            
            if len(valid_indices) > 0:
                # Check each valid configuration for collisions
                for idx in valid_indices:
                    config = configs[idx]
                    config_tensor = torch.tensor(config, device=device, dtype=torch.float32).unsqueeze(0)
                    
                    # Check collision
                    sdf_values = robot_sdf.query_sdf(
                        points=points_robot.unsqueeze(0),
                        joint_angles=config_tensor,
                        return_gradients=False
                    )
                    
                    if sdf_values.min() > 0.02:  # Safety margin
                        current_dist = distances[idx].item()
                        if current_dist < best_dist:
                            best_dist = current_dist
                            best_config = config
                            print(f"Found better configuration:")
                            print(f"Distance to goal: {best_dist:.4f} meters")
                            
                            # If distance is very good, we can return early
                            if best_dist < threshold * 0.5:
                                return best_config, best_dist
                
        except Exception as e:
            print(f"Error in batch {batch}: {str(e)}")
            continue
            
        if (batch + 1) % 10 == 0:
            print(f"Processed {(batch + 1) * batch_size}/{n_samples} samples...")
    
    if best_config is not None:
        print(f"Found goal configuration with distance: {best_dist:.4f} meters")
        return best_config, best_dist
    else:
        print("Failed to find valid goal configuration")
        return None, float('inf')
    
def load_point_cloud(device: str = 'cuda') -> torch.Tensor:
    """Load the filtered point cloud from the default location.
    
    Args:
        device: Device to load the points to
        
    Returns:
        torch.Tensor: Point cloud tensor [N, 3]
    """
    # Get the path to the point cloud file
    current_dir = Path(__file__).parent.parent
    pointcloud_path = current_dir / "pointcloud_data" / "filtered_final_points.npy"
    
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"Point cloud file not found at {pointcloud_path}")
    
    # Load and convert to tensor
    points = np.load(pointcloud_path)
    return torch.tensor(points, device=device, dtype=torch.float32)

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example goal position (modify as needed)
    goal_pos = torch.tensor([0.1742, 0.4552, 0.6078], device=device)
    
    try:
        # Load point cloud
        print("Loading point cloud...")
        points_robot = load_point_cloud(device)
        print(f"Loaded {len(points_robot)} points")
        
        # Initialize models
        print("Initializing robot models...")
        robot_fk = XArmFK(device=device)
        robot_sdf = RobotSDF(device=device)
        
        # Find goal configuration
        goal_config, achieved_distance = find_goal_configuration(
            goal_pos=goal_pos,
            points_robot=points_robot,
            robot_fk=robot_fk,
            robot_sdf=robot_sdf,
            n_samples=100000,
            threshold=0.1,
            device=device
        )
        
        if goal_config is not None:
            print("\nSuccess!")
            print("Found goal configuration (in radians):", goal_config)
            print("Found goal configuration (in degrees):", np.degrees(goal_config))
            print(f"Distance to goal: {achieved_distance:.4f} meters")
            
            # Save the configuration
            save_path = Path(__file__).parent / "goal_config.npy"
            np.save(save_path, goal_config)
            print(f"\nSaved goal configuration to: {save_path}")
        else:
            print("\nFailed to find a valid configuration")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()