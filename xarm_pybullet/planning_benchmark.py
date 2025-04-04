import torch
import numpy as np
from xarm_planning import XArmSDFVisualizer
from typing import List, Dict, Tuple
import json
from pathlib import Path
from dataclasses import dataclass
import cProfile
import pstats
from pstats import SortKey

@dataclass
class PlanningProgress:
    collision_checks: List[int]
    path_lengths: List[float]
    timestamps: List[float]  # Optional: if we want to show time progression

@dataclass
class PlanningMetrics:
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float
    min_workspace_cdf: float
    min_self_collision_cdf: float

def generate_random_goals(num_goals: int, seed: int = 42) -> List[torch.Tensor]:
    """Generate random reachable goal positions for the xArm"""
    np.random.seed(seed)
    
    # Define workspace bounds (in meters)
    x_range = (0.0, 0.6)    # Forward/backward
    y_range = (-0.3, 0.3)   # Left/right
    z_range = (0.3, 0.8)    # Up/down
    
    goals = []
    for _ in range(num_goals):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)
        goals.append(torch.tensor([x, y, z], device='cuda'))
    
    return goals

def find_goal_configs(goal: torch.Tensor, seed: int) -> List[np.ndarray]:
    """Find goal configurations for a given target position"""
    # Use any planner type since we only need the goal config finding functionality
    visualizer = XArmSDFVisualizer(
        goal,
        use_gui=False,
        planner_type='cdf_rrt',
        seed=seed,
        use_pybullet_inverse=True
    )
    return visualizer._find_goal_configuration(goal)

def run_benchmark(num_trials: int = 10, base_seed: int = 42, use_profile: bool = False):
    # Create a profiler only if needed
    profiler = cProfile.Profile() if use_profile else None
    
    # Define planners to test
    planners = ['bubble'] #, ['cdf_rrt', 'sdf_rrt', 'lazy_rrt', 'rrt_connect'], ['bubble', 'bubble_connect']
    results = {planner: {
        'collision_checks': [],
        'path_lengths': [],
        'planning_times': [],
        'min_workspace_cdfs': [],
        'min_self_collision_cdfs': [],
        'successes': 0
    } for planner in planners}
    
    goals = generate_random_goals(num_trials, seed=base_seed)
    goal_configs = {}

    print("Finding goal configurations...")
    for i, goal in enumerate(goals):
        print(f"\nGoal {i+1}/{num_trials}")
        print(f"Goal position: {goal.cpu().numpy()}")
        
        # Create one visualizer instance for this goal
        if planners[0] == 'bubble':
            planner_type = 'bubble'
        else:
            planner_type = 'cdf_rrt'

        visualizer = XArmSDFVisualizer(
            goal,
            use_gui=False,
            planner_type=planner_type,  # Initial planner 
            seed=base_seed,
            use_pybullet_inverse=True,
            early_termination=True
        )
        
        try:
            configs = visualizer._find_goal_configuration(goal, threshold=0.08)
            if configs:
                goal_configs[i] = configs
                print(f"Found {len(configs)} goal configurations")
                
                # Test each planner using the same visualizer instance
                for planner in planners:
                    print(f"\n{'='*50}")
                    print(f"Testing {planner.upper()} planner for goal {i+1}")
                    print(f"{'='*50}")
                    
                    # Update planner type and configurations
                    visualizer.planner_type = planner
                    visualizer.goal_configs = configs
                    visualizer._found_goal_configs = True
                    
                    try:
                        # Start profiling before planning if enabled
                        if use_profile:
                            profiler.enable()
                        
                        # Run planning without execution
                        result = visualizer.run_demo(execute_trajectory=False)
                        
                        # Stop profiling after planning if enabled
                        if use_profile:
                            profiler.disable()
                            # Print profiling results
                            stats = pstats.Stats(profiler).sort_stats(SortKey.TIME)
                            print(f"\nProfiling results for {planner}:")
                            stats.print_stats(20)  # Print top 20 time-consuming functions
                        
                        if result is not None:
                            metrics = result['metrics'] if isinstance(result, dict) else result
                            results[planner]['collision_checks'].append(metrics.num_collision_checks)
                            results[planner]['path_lengths'].append(metrics.path_length)
                            results[planner]['planning_times'].append(metrics.planning_time)
                            
                            # Add tracking of minimum CDF values along path
                            if 'waypoints' in result:
                                waypoints = result['waypoints']
                                min_workspace_cdf = float('inf')
                                min_self_collision_cdf = float('inf')
                                
                                for config in waypoints:
                                    # Check workspace CDF
                                    workspace_cdf = visualizer.robot_cdf.query_cdf(
                                        points=visualizer.points_robot.unsqueeze(0),
                                        joint_angles=torch.tensor(config, device=visualizer.device).unsqueeze(0),
                                        return_gradients=False
                                    ).min().item()
                                    
                                    # Check self-collision CDF
                                    self_collision_cdf = visualizer.self_collision_cdf.get_cdf(config)
                                    
                                    min_workspace_cdf = min(min_workspace_cdf, workspace_cdf)
                                    min_self_collision_cdf = min(min_self_collision_cdf, self_collision_cdf)
                                
                                results[planner]['min_workspace_cdfs'].append(min_workspace_cdf)
                                results[planner]['min_self_collision_cdfs'].append(min_self_collision_cdf)
                            
                            results[planner]['successes'] += 1
                            print(f"Success - Collision checks: {metrics.num_collision_checks}")
                            print(f"Path length: {metrics.path_length:.3f}")
                            print(f"Planning time: {metrics.planning_time:.3f}s")
                        else:
                            print("Planning failed")
                    except Exception as e:
                        print(f"Error during planning with {planner}: {str(e)}")
            else:
                print("No valid goal configurations found")
        except Exception as e:
            print(f"Error finding goal configurations: {str(e)}")
        finally:
            # Clean up visualizer only after testing all planners for this goal
            if hasattr(visualizer, 'env'):
                visualizer.env.close()

    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    valid_trials = len(goal_configs)
    print(f"\nFound valid goal configurations for {valid_trials}/{num_trials} targets")
    
    if valid_trials == 0:
        print("No valid trials completed. Try adjusting the workspace bounds.")
        return results
    
    for planner, data in results.items():
        success_rate = data['successes'] / valid_trials if valid_trials > 0 else 0
        print(f"\n{planner.upper()}:")
        print(f"Success Rate: {success_rate*100:.1f}%")
        
        if data['successes'] > 0:
            print(f"Collision Checks: {np.mean(data['collision_checks']):.1f} ± {np.std(data['collision_checks']):.1f}")
            print(f"Path Length: {np.mean(data['path_lengths']):.3f} ± {np.std(data['path_lengths']):.3f}")
            print(f"Planning Time: {np.mean(data['planning_times']):.3f}s ± {np.std(data['planning_times']):.3f}s")
            print(f"Min Workspace CDF: {np.mean(data['min_workspace_cdfs']):.3f} ± {np.std(data['min_workspace_cdfs']):.3f}")
            print(f"Min Self-collision CDF: {np.mean(data['min_self_collision_cdfs']):.3f} ± {np.std(data['min_self_collision_cdfs']):.3f}")
    
    # Save results
    Path("benchmark_results").mkdir(exist_ok=True)
    with open("benchmark_results/benchmark_summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results

if __name__ == "__main__":
    results = run_benchmark(num_trials=20, base_seed=5, use_profile=False) 