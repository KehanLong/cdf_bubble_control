import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import argparse

from utils_env import create_obstacles
from robot_cdf import RobotCDF
from robot_sdf import RobotSDF
from utils_new import inverse_kinematics_analytical

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))



def parse_args():
    parser = argparse.ArgumentParser(description='Planning Benchmark')
    parser.add_argument('--planner_type', type=str, default='rrt',
                      choices=['rrt', 'bubble'],
                      help='Type of planner to use (default: rrt)')
    parser.add_argument('--num_envs', type=int, default=100,
                      help='Number of environments to test (default: 100)')
    parser.add_argument('--seed', type=int, default=2,
                      help='Random seed (default: 2)')
    parser.add_argument('--early_termination', type=bool, default=True,
                      help='Whether to terminate early when solution is found (default: True)')
    parser.add_argument('--one_goal', type=bool, default=False,
                      help='Whether to use single goal configuration (default: False)')
    return parser.parse_args()


def run_planning_benchmark(planner_type: str = "rrt", num_envs: int = 100, 
                         seed: int = 3, early_termination: bool = True, 
                         one_goal: bool = False):
    """Run planning benchmark across multiple environments and planners."""
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    # Define goal positions
    # goal_positions = np.array([
    #     [0, 3], [0, 3.5], [-0.5, 3], [-1, 3], [-1.5, 2.5],
    #     [-2, -2], [-2.5, 1.5], [-3, 1],  [-3.5, 0],
    #     [-3.5, -0.5], [-3, -1], [-2.5, -0.5], [-2, -1], [-2, -1.5],
    #     [-0.5, -3], [0, -3]
    # ])

    goal_positions = np.array([
        [0, 3], [0, 3.5], [-3, 1], [-2.5, -0.5]
    ])

    
    
    # Initialize results storage
    results = []
    
    # Initialize robot models
    robot_cdf = RobotCDF(device=device)
    # Initialize SDF model for RRT-based planners
    robot_sdf = RobotSDF(device=device) if planner_type != "bubble" else None
    
    # Setup joint limits
    initial_config = np.array([0., 0.], dtype=np.float32)
    joint_limits = (
        np.full_like(initial_config, -np.pi),  # lower bounds
        np.full_like(initial_config, np.pi)    # upper bounds
    )
    
    # Setup planners based on type
    if planner_type == "bubble":
        from planner.bubble_planner import BubblePlanner
        planners = {
            'bubble': BubblePlanner(
                robot_cdf, joint_limits, max_samples=300, batch_size=2,
                device=device, seed=seed, early_termination=early_termination
            ),
            'bubble_connect': BubblePlanner(
                robot_cdf, joint_limits, max_samples=300, batch_size=2,
                device=device, seed=seed, early_termination=early_termination, planner_type='bubble_connect'
            )
        }
    else:  # rrt
        from planner.rrt_ompl import OMPLRRTPlanner
        planners = {
            'cdf_rrt': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='cdf_rrt'
            ),
            'sdf_rrt': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='sdf_rrt'
            ),
            'lazy_rrt': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='lazy_rrt'
            ),
            'rrt_connect': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='rrt_connect'
            ),
        }
    
    # Run experiments
    for env_idx in tqdm(range(num_envs), desc="Testing environments"):
        # Create random environment
        rng = np.random.default_rng(env_idx + seed)  
        obstacles = create_obstacles(rng=rng)
        obstacle_points = torch.tensor(np.concatenate(obstacles, axis=0), device=device)
        
        # Randomly select a goal position
        goal_idx = rng.integers(0, len(goal_positions))
        goal_pos = goal_positions[goal_idx]

        print(f"Goal position: {goal_pos}")
        
        # Get goal configurations using IK
        goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])
        
        # Convert goal configs to tensor and check CDF values
        goal_configs_tensor = torch.tensor(np.stack(goal_configs), device=device)
        cdf_values = robot_cdf.query_cdf(obstacle_points.unsqueeze(0).expand(len(goal_configs), -1, -1), 
                                        goal_configs_tensor)

        # Skip trial if any goal configuration has CDF value below threshold
        if torch.any(cdf_values.min(dim=1)[0] < 0.1):
            print(f"Skipping trial - found goal configuration with CDF value below threshold")
            continue

        if one_goal:
            goal_configs = np.array([goal_configs[rng.integers(0, len(goal_configs))]])  # Keep as 2D array
        
        
        # Test planners
        for planner_name, planner in planners.items():
            try:
                # Plan with early termination (single goal case)
                if planner_type == "bubble":
                    result = planner.plan(
                        initial_config, goal_configs, obstacle_points
                    )
                else:
                    result = planner.plan(
                        start_config=initial_config,
                        goal_configs=goal_configs,
                        obstacle_points=obstacle_points,
                        max_time=10.0,
                        early_termination=early_termination
                    )
            
    
                # Store results
                if result is not None and result['metrics'].success:
                    results.append({
                        'env_idx': env_idx,
                        'planner': planner_name,
                        'num_collision_checks': result['metrics'].num_collision_checks,
                        'path_length': result['metrics'].path_length,
                        'planning_time': result['metrics'].planning_time,
                        'success': True
                    })
                else:
                    results.append({
                        'env_idx': env_idx,
                        'planner': planner_name,
                        'num_collision_checks': float('inf'),
                        'path_length': float('inf'),
                        'planning_time': float('inf'),
                        'success': False
                    })
            
            except Exception as e:
                print(f"Error with planner {planner_name} on env {env_idx}: {str(e)}")
                results.append({
                    'env_idx': env_idx,
                    'planner': planner_name,
                    'num_collision_checks': float('inf'),
                    'path_length': float('inf'),
                    'planning_time': float('inf'),
                    'success': False
                })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics
    stats = df.groupby('planner').agg({
        'num_collision_checks': ['mean', 'std'],
        'path_length': ['mean', 'std'],
        'planning_time': ['mean', 'std'],
        'success': 'mean'
    }).round(2)
    
    print(f"\n{planner_type.upper()} Planning Statistics:")
    print(stats)
    
    # Save results (commented out)
    # df.to_csv(f'{planner_type}_planning_results.csv', index=False)
    # stats.to_csv(f'{planner_type}_planning_stats.csv')
    
    return df, stats

if __name__ == "__main__":
    args = parse_args()
    df, stats = run_planning_benchmark(
        planner_type=args.planner_type,
        num_envs=args.num_envs,
        seed=args.seed,
        early_termination=args.early_termination,
        one_goal=args.one_goal
    )