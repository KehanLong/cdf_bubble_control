import numpy as np
from typing import List
from utils_env import create_obstacles, plot_environment
from utils_visualization import visualize_results, plot_path_comparison
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import torch
import matplotlib.pyplot as plt
import imageio

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from planner.bubble_planner import BubblePlanner, PlanningMetrics

from control.clf_cbf_qp import ClfCbfQpController
from control.clf_dro_cbf import ClfCbfDrccpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
from control.reference_governor_bezier import BezierReferenceGovernor
import jax
import jax.numpy as jnp

from utils_new import compute_robot_distances

src_dir = os.path.dirname(os.path.abspath(__file__))


class CDF:
    def __init__(self, weights_file, obstacles, device, truncation_value=0.3):
        self.model = load_learned_cdf(weights_file)
        self.model.to(device)
        self.obstacle_points = concatenate_obstacle_list(obstacles)
        self.device = device
        self.truncation_value = truncation_value
    
    def __call__(self, configurations):
        cdf_values = cdf_evaluate_model(self.model, configurations, self.obstacle_points, self.device)
        min_cdfs = np.min(cdf_values, axis=1)
        # Truncate values larger than truncation_value
        return np.minimum(min_cdfs, self.truncation_value)


def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
    obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
    np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)

    # plt.show()

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

def plan_and_visualize(cdf, obstacles, initial_config, goal_config):
    """Plan and visualize a path using the bubble planner"""
    print("\nStarting planning process...")
    # Set random seed for reproducibility
    seed = 43
    
    # Create RobotCDF wrapper
    robot_cdf = RobotCDF(cdf.model, cdf.device, cdf.truncation_value)
    
    # Initialize bubble planner
    joint_limits = (
        np.full_like(initial_config, -np.pi),  # lower bounds
        np.full_like(initial_config, np.pi)    # upper bounds
    )
    planner = BubblePlanner(robot_cdf, joint_limits, device=cdf.device, seed=seed)
    
    # Get obstacle points
    obstacle_points = torch.tensor(concatenate_obstacle_list(obstacles), 
                                 device=cdf.device)
    print(f"Obstacle points shape: {obstacle_points.shape}")
    
    # Plan path
    result = planner.plan(initial_config, goal_config, obstacle_points)
    
    if result is None:
        print("Planning failed!")
        return None, None
    
    # Extract trajectory and metrics
    trajectory = result['waypoints']
    metrics = result['metrics']
    
    print(f"\nPlanning metrics:")
    print(f"Success: {metrics.success}")
    print(f"Planning time: {metrics.planning_time:.2f} seconds")
    print(f"Number of samples: {metrics.num_samples}")
    print(f"Path length: {metrics.path_length:.2f}")
    print(f"Number of collision checks: {metrics.num_collision_checks}")
    
    # Create visualization
    visualize_results(obstacles, initial_config, goal_config, 
                     trajectory, result['bezier_curves'], src_dir = src_dir)
    
    # Create animation if trajectory exists
    if trajectory is not None:
        animate_path(obstacles, result, fps=10, duration=15.0, src_dir = src_dir)
    
    return result


def track_planned_path(obstacles, trajectory_data, initial_config, dt=0.02, duration=20.0, 
                      control_type='clf_dro_cbf', use_bezier=True):
    """
    Track the planned path using selected controller and reference governor.
    
    Args:
        obstacles: List of obstacle arrays
        trajectory_data: Dictionary containing trajectory information
            - waypoints: Discrete waypoints
            - bezier_curves: Bezier curve segments (optional)
            - times: Time parameterization
        initial_config: Initial joint configuration
        dt: Time step
        duration: Maximum duration
        control_type: Controller type ('pd', 'clf_cbf', or 'clf_dro_cbf')
        use_bezier: Whether to use Bezier curves (True) or discrete waypoints (False)
    """
    # Convert obstacles to JAX array first
    obstacle_points = jnp.array(concatenate_obstacle_list(obstacles))
    
    # Initialize appropriate controller based on type
    if control_type == 'pd':
        controller = PDController(kp=0.8, kd=0.1, control_limits=2.0)
    elif control_type == 'clf_cbf':
        controller = ClfCbfQpController(
            p1=1.0,    
            p2=1e2,    # Increased penalty on CLF slack
            clf_rate=2.0,  # Increased CLF convergence rate
            cbf_rate=1.0,
            safety_margin=0.05,
            state_dim=2,
            control_limits=2.0
        )
    elif control_type == 'clf_dro_cbf':
        controller = ClfCbfDrccpController(
            p1=1.0, 
            p2=1e1, 
            clf_rate=1.0, 
            cbf_rate=1.0, 
            wasserstein_r=0.01, 
            epsilon=0.1, 
            num_samples=5,
            state_dim=2,
            control_limits=2.0
        )
    
    # Initialize appropriate governor based on mode
    if use_bezier and 'bezier_curves' in trajectory_data:
        print("Using Bezier-based reference governor")
        governor = BezierReferenceGovernor(
            initial_state=initial_config,
            trajectory_data=trajectory_data,
            dt=dt,
            k=0.2,  # Slower progression
            zeta=8   # Smoother progression
        )
    else:
        print("Using discrete waypoint-based reference governor")
        waypoints = (trajectory_data['waypoints'] if isinstance(trajectory_data, dict) 
                    else trajectory_data)
        governor = ReferenceGovernor(
            initial_state=initial_config,
            path_configs=waypoints,
            dt=dt
        )
    
    # Initialize simulation
    current_config = initial_config
    current_vel = np.zeros_like(initial_config)
    tracked_configs = [current_config]
    reference_configs = [current_config]
    time = 0
    
    # Compute distances and gradients using JAX functions
    compute_distances = jax.jit(compute_robot_distances)
    
    while time < duration:
        # Get reference from governor
        if use_bezier and 'bezier_curves' in trajectory_data:
            reference_config, s, reference_vel = governor.update(current_config)
        else:
            reference_config, s = governor.update(current_config)
            
        # Store reference
        reference_configs.append(reference_config)
        
        # Generate control based on controller type
        if control_type == 'pd':
            u = controller.compute_control(
                current_config,
                reference_config,
                current_vel
            )
        else:  # clf_cbf or clf_dro_cbf
            # Get all distances
            distances = compute_distances(jnp.array(current_config), obstacle_points)
            
            if control_type == 'clf_cbf':
                # Get minimum distance and its gradient
                h = float(jnp.min(distances))
                dh_dtheta = jax.grad(lambda q: jnp.min(compute_distances(q, obstacle_points)))(
                    jnp.array(current_config)
                )
                
                u = controller.generate_controller(
                    current_config,
                    reference_config,
                    h,
                    np.array(dh_dtheta),
                    0.0  # Static obstacles, so dh/dt = 0
                )
            else:  # clf_dro_cbf
                # Get k smallest distances and their gradients
                k = controller.num_samples
                h_values = jnp.sort(distances)[:k]
                
                h_grads = []
                for i in range(k):
                    def get_ith_smallest(q, i=i):
                        dists = compute_distances(q, obstacle_points)
                        return jnp.sort(dists)[i]
                    
                    grad_i = jax.grad(get_ith_smallest)(jnp.array(current_config))
                    h_grads.append(np.array(grad_i))
                
                u = controller.generate_controller(
                    current_config,
                    reference_config,
                    np.array(h_values),
                    np.array(h_grads),
                    np.zeros(k)  # Static obstacles, so dh/dt = 0
                )
        
        # Simple forward dynamics
        current_vel = u
        current_config = current_config + u * dt
        
        # Store result
        tracked_configs.append(current_config)
        
        # Update time
        time += dt
        
        # Print progress occasionally
        if int(time/dt) % 50 == 0:
            tracking_error = np.linalg.norm(current_config - reference_config)
            print(f"Time: {time:.1f}s, s: {s:.3f}, Tracking error: {tracking_error:.3f}")
        
        # Break if we're close to the end AND tracking error is small
        tracking_error = np.linalg.norm(current_config - reference_config)
        if s > 0.99 and tracking_error < 0.05:
            print(f"\nReached goal at time {time:.1f}s")
            print(f"Final tracking error: {tracking_error:.3f}")
            break
    
    if time >= duration:
        print(f"\nReached duration limit ({duration}s)")
        print(f"Final s: {s:.3f}")
        print(f"Final tracking error: {tracking_error:.3f}")
    
    return np.array(tracked_configs), np.array(reference_configs)

def animate_path(obstacles: List[np.ndarray], trajectory_data, fps: int = 10, duration: float = 20.0, src_dir=src_dir):
    """
    Create an animation of the robot arm tracking the planned path.
    """
    print(f"\nStarting animation with duration: {duration}s")
    
    # Get tracked configurations and reference configs
    tracked_configs, reference_configs = track_planned_path(
        obstacles, 
        trajectory_data, 
        trajectory_data['waypoints'][0],
        dt=0.02, 
        duration=duration,  # Increased duration
        control_type='clf_cbf',
        use_bezier=True
    )
    
    # Create comparison plot
    plot_path_comparison(trajectory_data['waypoints'], tracked_configs, src_dir = src_dir)
    
    n_frames = int(fps * duration)
    path_indices = np.linspace(0, len(tracked_configs) - 1, n_frames, dtype=int)
    
    
    # Create figure once
    fig, ax = plt.subplots(figsize=(10, 10))
    frames = []
    
    for i in path_indices:
        ax.clear()
        current_config = tracked_configs[i]
        
        
        # Plot the environment and current configuration
        plot_environment(obstacles, current_config, ax=ax, robot_color='blue', label='Current')
        
        # Plot reference configuration
        plot_environment(obstacles, reference_configs[i], ax=ax, robot_color='green', 
                        plot_obstacles=False, label='Reference', robot_alpha=0.5)
        
        # Set consistent axis limits
        ax.set_title(f'Frame {i}/{len(path_indices)}')
        ax.legend()
        
        # Convert plot to RGB array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    # Save animation
    print("Saving animation...")
    imageio.mimsave(os.path.join(src_dir, 'figures/robot_animation.mp4'), frames, fps=fps)
    print("Animation saved as 'figures/robot_animation.mp4'")


if __name__ == "__main__":
    __spec__ = None

    trained_model_path = os.path.join(src_dir, "trained_models/cdf_models/cdf_model_2_links_truncated_new.pt")  # Adjust path as needed, 2_links or 4_links
    torch_model = load_learned_cdf(trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(12345)

    # Create obstacles
    obstacles = create_obstacles(rng=rng)

    cdf = CDF(trained_model_path, obstacles, device, truncation_value=0.4)


    # Make sure the figures directory exists
    os.makedirs(os.path.join(src_dir, 'figures'), exist_ok=True)

    # Set initial and goal configurations
    initial_config = np.array([0, 0])
    goal_config = np.array([2., 2.])

    # Plan and visualize
    result = plan_and_visualize(cdf, obstacles, initial_config, goal_config)
    plt.show()
