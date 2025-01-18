import numpy as np
from typing import List
from utils_env import create_obstacles
from utils_visualization import visualize_results, visualize_cdf_planning
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import torch

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from planner.bubble_planner import BubblePlanner, PlanningMetrics
# from planner.rrt_ompl import OMPLRRTPlanner



src_dir = os.path.dirname(os.path.abspath(__file__))


def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
    obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
    np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)

# a wrapper class for cdf model
class RobotCDF:
    def __init__(self, model, device, truncation_value=0.3):
        self.model = model
        self.device = device
        self.truncation_value = truncation_value
    
    def query_cdf(self, points: torch.Tensor, joint_angles: torch.Tensor, return_gradients: bool = False) -> torch.Tensor:
        """
        Query CDF values for given points and joint angles.
        """
        
        # Ensure points and joint_angles are properly shaped
        if points.dim() == 4:  # [1, 1, N, 2]
            points = points.squeeze(1)  # [1, N, 2]
        
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)  # Add batch dimension
            
        # Convert to numpy for cdf_evaluate_model
        points_np = points.cpu().numpy()
        joint_angles_np = joint_angles.cpu().numpy()
        
        # Get CDF values
        cdf_values = cdf_evaluate_model(self.model, joint_angles_np, points_np, self.device)
        
        # Convert back to tensor
        cdf_values = torch.tensor(cdf_values, device=self.device)
        
        # Truncate values larger than truncation_value
        cdf_values = torch.clamp(cdf_values, max=self.truncation_value)
        
        return cdf_values

def plan_and_visualize(robot_cdf, obstacles, initial_config, goal_configs, max_bubble_samples=500):
    """Plan and visualize a path using the bubble planner"""
    print("\nStarting planning process...")
    # Set random seed for reproducibility
    seed = 42
    
    # Initialize bubble planner
    joint_limits = (
        np.full_like(initial_config, -np.pi),  # lower bounds
        np.full_like(initial_config, np.pi)    # upper bounds
    )
    planner = BubblePlanner(robot_cdf, joint_limits, max_samples=max_bubble_samples, device=robot_cdf.device, seed=seed, planner_type='bubble')
    
    # Get obstacle points
    obstacle_points = torch.tensor(concatenate_obstacle_list(obstacles), 
                                 device=robot_cdf.device)
    print(f"Obstacle points shape: {obstacle_points.shape}")
    
    # Plan path (goal_configs is already a list of numpy arrays)
    result = planner.plan(initial_config, goal_configs, obstacle_points)
    
    if result is None:
        print("Planning failed!")
        return None
    
    # Extract trajectory and metrics
    trajectory = result['waypoints']
    metrics = result['metrics']
    
    print(f"\nPlanning metrics:")
    print(f"Success: {metrics.success}")
    print(f"Planning time: {metrics.planning_time:.2f} seconds")
    print(f"Number of samples: {metrics.num_samples}")
    print(f"Path length: {metrics.path_length:.2f}")
    print(f"Number of collision checks: {metrics.num_collision_checks}")
    print(f"Reached goal index: {metrics.reached_goal_index}")
    # Create visualization
    visualize_results(obstacles, initial_config, goal_configs[metrics.reached_goal_index], 
                     trajectory, result['bezier_curves'], src_dir=src_dir)
    
    # Add CDF planning visualization
    visualize_cdf_planning(robot_cdf, initial_config, goal_configs, 
                         trajectory, result['bubbles'], 
                         obstacle_points, src_dir)
    
    return result

if __name__ == "__main__":


    trained_model_path = os.path.join(src_dir, "trained_models/cdf_models/cdf_model_2_links_truncated_new.pt")  # Adjust path as needed
    torch_model = load_learned_cdf(trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(12345)

    # Create obstacles
    obstacles = create_obstacles(rng=rng)

    robot_cdf = RobotCDF(torch_model, device, truncation_value=0.3)
    # Make sure the figures directory exists
    os.makedirs(os.path.join(src_dir, 'figures'), exist_ok=True)

    # Set initial and goal configurations
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_configs = [
        np.array([2., 2.], dtype=np.float32),     # First goal
        np.array([1., -2.], dtype=np.float32)     # Second goal
    ]

    # Plan and visualize with multiple goals
    result = plan_and_visualize(robot_cdf, obstacles, initial_config, goal_configs)

    # Save planning result for control script
    # if result is not None:
    #     np.save(os.path.join(src_dir, 'planned_path.npy'), result)

