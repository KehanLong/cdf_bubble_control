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
from control.clf_cbf_qp import ClfCbfQpController
from control.clf_dro_cbf import ClfCbfDrccpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
from control.reference_governor_bezier import BezierReferenceGovernor
from main_planning import plan_and_visualize, concatenate_obstacle_list
from utils_env import create_obstacles

from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from utils_new import inverse_kinematics_analytical



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
    # Convert obstacles to tensor (keep as 2D points)
    obstacle_points = torch.tensor(concatenate_obstacle_list(obstacles), dtype=torch.float32)  # Shape: [N, 2]
    
    # Initialize RobotCDF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cdf = RobotCDF(device)
    obstacle_points = obstacle_points.to(device)
    
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
            p2=1e2, 
            clf_rate=2.0, 
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
            zeta=10   # Smoother progression
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
            # Convert current config to appropriate shape for RobotCDF and enable gradients
            config_tensor = torch.tensor(current_config, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: [1, 2]
            config_tensor.requires_grad_(True)
            
            # Get CDF values (without gradients flag)
            cdf_values = robot_cdf.query_cdf(
                points=obstacle_points.unsqueeze(0),  # Shape: [1, N, 2]
                joint_angles=config_tensor,           # Shape: [1, 2]
                return_gradients=False
            )
            
            if control_type == 'clf_cbf':
                # Get minimum CDF value
                h = float(torch.min(cdf_values).item())   # Adding safety margin
                min_idx = torch.argmin(cdf_values)
                
                # Compute gradient for minimum value
                h_val = cdf_values[0, min_idx]
                h_val.backward()
                dh_dtheta = config_tensor.grad.clone().cpu().numpy()[0]
                config_tensor.grad.zero_()
                
                u = controller.generate_controller(
                    current_config,
                    reference_config,
                    h,
                    dh_dtheta,
                    0.0  # Static obstacles, so dh/dt = 0
                )
            else:  # clf_dro_cbf
                # Get k smallest CDF values
                k = controller.num_samples
                h_values, indices = torch.topk(cdf_values[0], k, largest=False)
                
                # Compute gradients for k smallest values
                h_grads = []
                for idx in indices:
                    h_val = cdf_values[0, idx]
                    h_val.backward(retain_graph=True)
                    h_grads.append(config_tensor.grad.clone().cpu().numpy()[0])
                    config_tensor.grad.zero_()
                
                u = controller.generate_controller(
                    current_config,
                    reference_config,
                    h_values.detach().cpu().numpy(),  # Adding safety margin
                    np.stack(h_grads),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create obstacles
    obstacles = create_obstacles(rng=rng)

    # Initialize RobotCDF and RobotSDF classes
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    # Make sure the figures directory exists
    os.makedirs(os.path.join(src_dir, 'figures'), exist_ok=True)

    # Set initial configuration and goal position
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_pos = np.array([-2.5, 2.5], dtype=np.float32)     # Single goal position for end-effector

    # find multiple goal configurations by inverse kinematics
    goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])

    # Test different planners: bubble, bubble_connect, cdf_rrt, sdf_rrt
    planner = 'bubble'
    result = plan_and_visualize(
            robot_cdf, robot_sdf, obstacles, initial_config, goal_configs, 
            max_bubble_samples=100, seed=seed, early_termination=False, 
            planner_type=planner
        )
    
    if result is None:
        print("Planning failed! Exiting...")
        sys.exit(1)
    
    # Execute control
    tracked_configs, reference_configs, tracked_vels, reference_vels = track_planned_path(
        obstacles, 
        result, 
        initial_config,
        dt=0.02, 
        duration=20.0,
        control_type='pd',                # clf_cbf, clf_dro_cbf, pd
        use_bezier=True
    )
    
    # Create animation
    animate_path(obstacles, tracked_configs, reference_configs, dt=0.02)