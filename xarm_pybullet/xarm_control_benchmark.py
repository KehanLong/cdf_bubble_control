import torch
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import directed_hausdorff

from xarm_planning import XArmSDFVisualizer
from xarm_control import XArmController

def compute_tracking_error(tracked_positions: np.ndarray, reference_positions: np.ndarray) -> float:
    """
    Compute tracking error as mean Euclidean distance between tracked and reference end-effector positions
    """
    return np.mean(np.linalg.norm(tracked_positions - reference_positions, axis=1))

def compute_frechet_distance(path1: np.ndarray, path2: np.ndarray) -> float:
    """
    Compute the Frechet distance between two paths using directed Hausdorff distance.
    """
    return max(directed_hausdorff(path1, path2)[0],
              directed_hausdorff(path2, path1)[0])

def plan_trajectory(goal_pos: torch.Tensor, seed: int) -> Optional[Dict]:
    """
    Plan a trajectory for a given goal position
    """
    try:
        # Initialize planner with no dynamic obstacles for planning
        planner = XArmSDFVisualizer(goal_pos, use_gui=False, planner_type='bubble', 
                                   dynamic_obstacles=False, seed=seed)
        
        # Plan trajectory
        trajectory_data = planner.run_demo(execute_trajectory=False)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in planning: {str(e)}")
        return None
    finally:
        if 'planner' in locals() and hasattr(planner, 'env'):
            planner.env.close()

def run_single_trial(goal_pos: torch.Tensor, trajectory_data: Dict, 
                    control_type: str, seed: int) -> Tuple[Optional[Dict], bool]:
    """
    Run a single trial for a specific controller type using pre-planned trajectory
    """
    try:
        # Initialize planner and controller with dynamic obstacles for execution
        planner = XArmSDFVisualizer(goal_pos, use_gui=False, planner_type='bubble', 
                                   dynamic_obstacles=False, seed=seed)
        controller = XArmController(planner, control_type=control_type)
        
        # Execute trajectory and collect metrics
        goal_distances, executed_configs, is_safe = controller.execute_trajectory(
            trajectory_data, 
            dt=0.02, 
            use_bezier=True
        )
        
        # Get planned configurations
        planned_configs = trajectory_data['waypoints']
        
        # Compute Frechet distance between planned and executed joint trajectories
        tracking_error = compute_frechet_distance(
            executed_configs,
            planned_configs
        )
        
        # Consider trial successful if both goal is reached AND execution was safe
        is_success = goal_distances[-1] < 0.1 and is_safe
        
        metrics = {
            'goal_distances': goal_distances,
            'final_distance': goal_distances[-1],
            'tracking_error': tracking_error,
            'is_success': is_success,
            'is_safe': is_safe
        }
        
        return metrics, True
        
    except Exception as e:
        print(f"Error in {control_type} trial: {str(e)}")
        return None, False
    finally:
        if 'controller' in locals() and hasattr(controller, 'planner'):
            controller.planner.env.close()

def run_benchmark(num_trials: int = 10, seed: int = 42) -> Dict:
    """
    Run benchmark comparing different control approaches
    """
    # Initialize results dictionary
    results = {
        'pd': {'success_rate': 0, 'safe_rate': 0, 'tracking_errors': []},
        'clf_cbf': {'success_rate': 0, 'safe_rate': 0, 'tracking_errors': []},
        'clf_dro_cbf': {'success_rate': 0, 'safe_rate': 0, 'tracking_errors': []}
    }
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Set random seed for reproducibility
        seed = trial + seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate random goal
        goal_pos = torch.tensor([
            np.random.uniform(0.3, 0.7),    # x
            np.random.uniform(-0.3, 0.3),   # y
            np.random.uniform(0.3, 0.8)     # z
        ], device='cuda')
        
        print(f"Testing goal position: {goal_pos.cpu().numpy()}")
        
        # Plan trajectory once for this goal
        trajectory_data = plan_trajectory(goal_pos, seed)
        
        if trajectory_data is None:
            print(f"Planning failed for goal, skipping trial")
            continue
        
        # Test each controller with the same planned trajectory
        for control_type in ['pd', 'clf_cbf', 'clf_dro_cbf']:
            print(f"\nTesting {control_type} controller...")
            
            metrics, completed = run_single_trial(goal_pos, trajectory_data, control_type, seed)
            
            if completed and metrics is not None:
                if metrics['is_success']:
                    results[control_type]['success_rate'] += 1
                if metrics['is_safe']:
                    results[control_type]['safe_rate'] += 1
                results[control_type]['tracking_errors'].append(metrics['tracking_error'])
                
                print(f"Success: {metrics['is_success']}")
                print(f"Safe: {metrics['is_safe']}")
                print(f"Final distance to goal: {metrics['final_distance']:.4f}")
                print(f"Trajectory tracking error: {metrics['tracking_error']:.4f}")
    
    # Compute final statistics
    for control_type in results:
        results[control_type]['success_rate'] /= num_trials
        results[control_type]['safe_rate'] /= num_trials
        if results[control_type]['tracking_errors']:
            results[control_type]['mean_tracking_error'] = np.mean(results[control_type]['tracking_errors'])
            results[control_type]['std_tracking_error'] = np.std(results[control_type]['tracking_errors'])
        else:
            results[control_type]['mean_tracking_error'] = float('nan')
            results[control_type]['std_tracking_error'] = float('nan')
    
    return results

def save_results(results: Dict, output_file: str):
    """
    Save benchmark results to a JSON file
    """
    serializable_results = {}
    for control_type, metrics in results.items():
        serializable_results[control_type] = {
            'success_rate': float(metrics['success_rate']),
            'safe_rate': float(metrics['safe_rate']),
            'mean_tracking_error': float(metrics['mean_tracking_error']),
            'std_tracking_error': float(metrics['std_tracking_error']),
            'tracking_errors': [float(x) for x in metrics['tracking_errors']]
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / 'benchmark_results'
    results_dir.mkdir(exist_ok=True)
    
    # Run benchmark
    print("Starting xArm control benchmark...")
    results = run_benchmark(num_trials=20, seed=42)
    
    # Save results
    # output_file = results_dir / 'xarm_control_benchmark_results.json'
    # save_results(results, str(output_file))
    
    # Print summary
    print("\nBenchmark Results:")
    print("-----------------")
    for control_type, metrics in results.items():
        print(f"\n{control_type.upper()}:")
        print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"Safety Rate: {metrics['safe_rate']*100:.1f}%")
        print(f"Mean Tracking Error: {metrics['mean_tracking_error']:.4f} ± {metrics['std_tracking_error']:.4f}")
    
    # print(f"\nDetailed results saved to: {output_file}") 