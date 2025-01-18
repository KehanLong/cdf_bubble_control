import numpy as np
import jax
import jax.numpy as jnp
import os
import sys
from pathlib import Path
from typing import List
import torch

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_dir = os.path.dirname(os.path.abspath(__file__))

from utils_visualization import create_animation
from utils_new import compute_robot_distances
from control.clf_cbf_qp import ClfCbfQpController
from control.clf_dro_cbf import ClfCbfDrccpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
from control.reference_governor_bezier import BezierReferenceGovernor
from main_planning import plan_and_visualize, RobotCDF, concatenate_obstacle_list
from cdf_evaluate import load_learned_cdf
from utils_env import create_obstacles



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
    # Convert obstacles to JAX array with float32 dtype
    obstacle_points = jnp.array(concatenate_obstacle_list(obstacles), dtype=jnp.float32)
    
    # Initialize appropriate controller based on type
    if control_type == 'pd':
        controller = PDController(kp=1.0, kd=0.1, control_limits=2.0)
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
    tracked_vels = [current_vel]
    reference_configs = [current_config]
    reference_vels = [np.zeros_like(initial_config)]
    time = 0
    
    # Compute distances and gradients using JAX functions
    compute_distances = jax.jit(compute_robot_distances)
    
    while time < duration:
        # Get reference from governor
        if use_bezier and 'bezier_curves' in trajectory_data:
            reference_config, s, reference_vel = governor.update(current_config)
        else:
            reference_config, s = governor.update(current_config)
            reference_vel = np.zeros_like(current_config)
            
        # Store reference
        reference_configs.append(reference_config)
        reference_vels.append(reference_vel)
        
        # Generate control based on controller type
        if control_type == 'pd':
            u = controller.compute_control(
                current_config,
                reference_config,
                current_vel
            )
        else:  # clf_cbf or clf_dro_cbf
            # Get all distances
            distances = compute_distances(jnp.array(current_config, dtype=jnp.float32), obstacle_points)
            
            if control_type == 'clf_cbf':
                # Get minimum distance and its gradient
                h = float(jnp.min(distances))
                dh_dtheta = jax.grad(lambda q: jnp.min(compute_distances(q, obstacle_points)))(
                    jnp.array(current_config, dtype=jnp.float32)  # Explicitly convert to float32
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
        
        # Proper dynamics integration (simple Euler integration)
        current_vel = u
        current_config = current_config + current_vel * dt
        
        # Store result
        tracked_configs.append(current_config)
        tracked_vels.append(current_vel)
        
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
    
    return np.array(tracked_configs), np.array(reference_configs), np.array(tracked_vels), np.array(reference_vels)

def animate_path(obstacles: List[np.ndarray], tracked_configs, reference_configs, dt: float = 0.02):
    """Create an animation of the robot arm tracking the planned path."""
    return create_animation(obstacles, tracked_configs, reference_configs, dt, src_dir)

if __name__ == "__main__":
    # Setup for planning
    trained_model_path = os.path.join(src_dir, "trained_models/cdf_models/cdf_model_2_links_truncated_new.pt")
    torch_model = load_learned_cdf(trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create obstacles and setup
    rng = np.random.default_rng(12345)
    obstacles = create_obstacles(rng=rng)
    robot_cdf = RobotCDF(torch_model, device, truncation_value=0.3)
    
    # Set configurations
    initial_config = np.array([0, 0])
    goal_config = np.array([2., 2.])
    
    # Plan path
    planned_path = plan_and_visualize(robot_cdf, obstacles, initial_config, goal_config)
    
    if planned_path is None:
        print("Planning failed! Exiting...")
        sys.exit(1)
    
    # Execute control
    tracked_configs, reference_configs, tracked_vels, reference_vels = track_planned_path(
        obstacles, 
        planned_path, 
        initial_config,
        dt=0.02, 
        duration=20.0,
        control_type='clf_cbf',
        use_bezier=True
    )
    
    # Create animation
    animate_path(obstacles, tracked_configs, reference_configs, dt=0.02)