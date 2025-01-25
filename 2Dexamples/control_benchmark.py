import numpy as np
import torch
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json
from scipy.spatial.distance import directed_hausdorff

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_dir = os.path.dirname(os.path.abspath(__file__))

from main_control import track_planned_path, generate_random_goal
from main_planning import plan_and_visualize
from utils_env import create_obstacles
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from utils_new import inverse_kinematics_analytical

def compute_frechet_distance(path1: np.ndarray, path2: np.ndarray) -> float:
    """
    Compute the Frechet distance between two paths using directed Hausdorff distance.
    """
    return max(directed_hausdorff(path1, path2)[0],
              directed_hausdorff(path2, path1)[0])


def run_benchmark(num_trials: int = 10) -> Dict:
    """
    Run benchmark comparing different control approaches.
    """
    # Initialize results dictionary
    results = {
        'pd': {'success_rate': 0, 'tracking_errors': []},
        'clf_cbf': {'success_rate': 0, 'tracking_errors': []},
        'clf_dro_cbf': {'success_rate': 0, 'tracking_errors': []}
    }
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Set random seed for reproducibility
        seed = trial
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        
        # Create random environment
        obstacles = create_obstacles(rng=rng)
        
        # Set initial and generate random goal configurations
        initial_config = np.array([0., 0.], dtype=np.float32)
        goal_pos = generate_random_goal(rng, robot_sdf, obstacles)
        goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])
        
        print(f"Goal position: {goal_pos}")
        
        # Plan path (without visualization)
        result = plan_and_visualize(
            robot_cdf, robot_sdf, obstacles, initial_config, goal_configs,
            max_bubble_samples=100, seed=seed, early_termination=False,
            planner_type='bubble', visualize=False  # Disable visualization
        )
        
        if result is None:
            print(f"Trial {trial + 1}: Planning failed, skipping...")
            continue
            
        # Test each controller
        for control_type in ['pd', 'clf_cbf', 'clf_dro_cbf']:
            print(f"\nTesting {control_type} controller...")
            
            tracked_configs, reference_configs, _, _, is_safe = track_planned_path(
                obstacles,
                result,
                initial_config,
                dt=0.02,
                duration=25.0,
                control_type=control_type,
                use_bezier=True,
                dynamic_obstacles_exist=True
            )
            
            if is_safe:
                results[control_type]['success_rate'] += 1
            
            # Compute tracking error (Frechet distance)
            tracking_error = compute_frechet_distance(tracked_configs, reference_configs)
            results[control_type]['tracking_errors'].append(tracking_error)
            
            print(f"Safe trajectory: {is_safe}")
            print(f"Tracking error: {tracking_error:.4f}")
    
    # Compute final statistics
    for control_type in results:
        results[control_type]['success_rate'] /= num_trials
        results[control_type]['mean_tracking_error'] = np.mean(results[control_type]['tracking_errors'])
        results[control_type]['std_tracking_error'] = np.std(results[control_type]['tracking_errors'])
    
    return results

def save_results(results: Dict, output_file: str):
    """
    Save benchmark results to a JSON file.
    """
    # Convert numpy types to Python native types for JSON serialization
    serializable_results = {}
    for control_type, metrics in results.items():
        serializable_results[control_type] = {
            'success_rate': float(metrics['success_rate']),
            'mean_tracking_error': float(metrics['mean_tracking_error']),
            'std_tracking_error': float(metrics['std_tracking_error']),
            'tracking_errors': [float(x) for x in metrics['tracking_errors']]
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = os.path.join(src_dir, 'benchmark_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run benchmark
    print("Starting control benchmark...")
    results = run_benchmark(num_trials=20)
    
    # Save results
    output_file = os.path.join(results_dir, 'control_benchmark_results.json')
    save_results(results, output_file)
    
    # Print summary
    print("\nBenchmark Results:")
    print("-----------------")
    for control_type, metrics in results.items():
        print(f"\n{control_type.upper()}:")
        print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"Mean Tracking Error: {metrics['mean_tracking_error']:.4f} ± {metrics['std_tracking_error']:.4f}")
    
    print(f"\nDetailed results saved to: {output_file}") 