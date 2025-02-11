import numpy as np
import os
import sys
from pathlib import Path
import torch
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_dir = os.path.dirname(os.path.abspath(__file__))

from main_planning import plan_and_visualize
from main_control import animate_path, create_dynamic_obstacles
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from utils_env import create_obstacles
from utils_new import inverse_kinematics_analytical
from control.pd_cbf_qp import ClfCbfQpController
from control.clf_dro_cbf import ClfCbfDrccpController
from control.pd_control import PDController
from control.reference_governor_bezier import BezierReferenceGovernor

def execute_planning_and_control(
    obstacles: List[np.ndarray],
    initial_config: np.ndarray,
    goal_pos: np.ndarray,
    control_type: str = 'clf_dro_cbf',
    planner_type: str = 'bubble',
    use_dynamic_obstacles: bool = False,
    dt: float = 0.02,
    goal_threshold: float = 0.05,
    safety_threshold: float = 0.01,
    max_duration: float = 40.0,
    initial_safety_margin: float = 0.1,  # Initial conservative margin
    replan_safety_margin: float = 0.2    # More conservative margin for replanning
) -> Dict[str, Any]:
    """
    Execute planning and control with continuous replanning based on controller feedback.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    # Convert static obstacles to tensor
    static_obstacle_points = torch.tensor(np.concatenate(obstacles), dtype=torch.float32, device=device)
    goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])
    
    # Initialize controllers
    pd_controller = PDController(kp=1.0, kd=0.2, control_limits=2.0)
    if control_type == 'clf_dro_cbf':
        controller = ClfCbfDrccpController(
            p1=1.0, p2=1e2, clf_rate=3.0, cbf_rate=1.0,
            wasserstein_r=0.015, epsilon=0.1, num_samples=10,
            state_dim=2, control_limits=5.0, stability_threshold=0.02
        )
    elif control_type == 'clf_cbf':
        controller = ClfCbfQpController(
            p1=1.0, p2=1e1, clf_rate=1.0, cbf_rate=1.0,
            safety_margin=0.2, state_dim=2, control_limits=2.0
        )
    else:
        controller = pd_controller
    
    # Initialize tracking lists
    tracked_configs = []
    reference_configs = []
    tracked_vels = []
    reference_vels = []
    
    # Initialize state
    current_config = initial_config
    current_vel = np.zeros_like(initial_config)
    time = 0.0
    need_replanning = True
    governor = None
    
    # Track if this is initial planning
    is_initial_plan = True
    
    while time < max_duration:
        if need_replanning:

            # Compute control input
            config_tensor = torch.tensor(current_config, dtype=torch.float32, device=device).unsqueeze(0)
            config_tensor.requires_grad_(True)
            print(f"\nPlanning from current config: {current_config}")
            print(f"currnet cdf value: {min(robot_cdf.query_cdf(static_obstacle_points.unsqueeze(0), config_tensor, return_gradients=False))}")
            
            # Use different safety margins for initial vs replanning
            current_safety_margin = initial_safety_margin if is_initial_plan else replan_safety_margin
            print(f"Using safety margin: {current_safety_margin}")
            
            # Plan new path
            result = plan_and_visualize(
                robot_cdf, robot_sdf, obstacles, current_config, goal_configs,
                max_bubble_samples=100, seed=int(time * 100),
                early_termination=False, planner_type=planner_type,
                visualize=True,
                safety_margin=current_safety_margin  # Pass safety margin to planner
            )
            
            if result is None:
                print("Planning failed!")
                break
            
            is_initial_plan = False  # Mark that initial planning is done
            
            # Initialize new reference governor
            governor = BezierReferenceGovernor(
                initial_state=current_config,
                trajectory_data=result,
                dt=dt,
                k=0.2,
                zeta=8
            )
            need_replanning = False
        
        # Get current dynamic obstacles if enabled
        if use_dynamic_obstacles:
            dynamic_obstacles, dynamic_velocities = create_dynamic_obstacles(time)
            dynamic_points = torch.tensor(np.concatenate(dynamic_obstacles), 
                                       dtype=torch.float32, device=device)
            dynamic_vels = torch.tensor(np.concatenate(dynamic_velocities), 
                                      dtype=torch.float32, device=device)
            all_obstacle_points = torch.cat([static_obstacle_points, dynamic_points], dim=0)
        else:
            all_obstacle_points = static_obstacle_points
            dynamic_vels = torch.zeros_like(all_obstacle_points)
        
        # Get reference from governor
        reference_config, s, reference_vel = governor.update(current_config)
        
        # Store states
        tracked_configs.append(current_config)
        reference_configs.append(reference_config)
        tracked_vels.append(current_vel)
        reference_vels.append(reference_vel)
        
        
        # Get CDF values
        cdf_values = robot_cdf.query_cdf(
            points=all_obstacle_points.unsqueeze(0),
            joint_angles=config_tensor,
            return_gradients=False
        )
        
        u_nominal = pd_controller.compute_control(
            current_config, reference_config, current_vel
        )
        
        # Compute control based on controller type
        if control_type == 'clf_dro_cbf':
            # Get k smallest CDF values and their gradients
            k = controller.num_samples
            h_values, indices = torch.topk(cdf_values[0], k, largest=False)
            h_grads = []
            dh_dt_values = []
            
            for idx in indices:
                h_val = cdf_values[0, idx]
                h_val.backward(retain_graph=True)
                h_grads.append(config_tensor.grad.clone().cpu().numpy()[0])
                config_tensor.grad.zero_()
                
                # Compute dh/dt for dynamic obstacles
                if use_dynamic_obstacles and idx >= len(static_obstacle_points):
                    points = all_obstacle_points.unsqueeze(0)
                    points.requires_grad_(True)
                    cdf_val = robot_cdf.query_cdf(points, config_tensor, return_gradients=False)[0, idx]
                    cdf_val.backward()
                    dh_dp = points.grad[0, idx]
                    dyn_idx = idx - len(static_obstacle_points)
                    dp_dt = dynamic_vels[dyn_idx]
                    dh_dt_values.append(torch.dot(dh_dp, dp_dt).item())
                else:
                    dh_dt_values.append(0.0)
            
            u = controller.generate_controller(
                current_config,
                reference_config,
                h_values.detach().cpu().numpy(),
                np.stack(h_grads),
                np.array(dh_dt_values),
                u_nominal
            )
            
            # Check controller's safety flag
            if controller.unstable_flag:
                print("Controller indicates unsafe condition (large CLF slack)")
                need_replanning = True
                continue
            
        else:
            u = u_nominal
        
        # Update state
        current_vel = u
        current_config = current_config + current_vel * dt
        time += dt
        
        # Check safety
        sdf_values = robot_sdf.query_sdf(all_obstacle_points.unsqueeze(0), config_tensor)
        if torch.min(sdf_values) <= safety_threshold:
            print(f"Safety violation detected at time {time:.2f}s")
            need_replanning = True
            continue
        
        # Check if goal is reached
        goal_error = min(np.linalg.norm(current_config - goal_configs[0]), np.linalg.norm(current_config - goal_configs[1]))
        if goal_error < goal_threshold:
            print(f"\nReached goal at time {time:.1f}s")
            print(f"Final goal error: {goal_error:.3f}")
            break
        
        # Print progress
        if int(time/dt) % 50 == 0:
            print(f"Time: {time:.1f}s, s: {s:.3f}, Goal error: {goal_error:.3f}")
    
    if time >= max_duration:
        print(f"\nReached maximum duration limit ({max_duration}s)")
        print(f"Final goal error: {goal_error:.3f}")
    
    # Create animation of the execution
    tracked_configs = np.array(tracked_configs)
    reference_configs = np.array(reference_configs)
    animate_path(obstacles, tracked_configs, reference_configs, 
                dt=dt, dynamic_obstacles=use_dynamic_obstacles)
    
    return {
        'success': goal_error < goal_threshold,
        'tracked_configs': tracked_configs,
        'reference_configs': reference_configs,
        'tracked_vels': np.array(tracked_vels),
        'reference_vels': np.array(reference_vels),
        'total_time': time,
        'timeout': time >= max_duration
    }

if __name__ == "__main__":
    # Setup
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rng = np.random.default_rng(seed)
    
    # Create obstacles
    obstacles = create_obstacles(rng=rng)
    
    # Set initial configuration and goal position
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_pos = np.array([-2.5, 2.5], dtype=np.float32)
    
    # Execute
    result = execute_planning_and_control(
        obstacles=obstacles,
        initial_config=initial_config,
        goal_pos=goal_pos,
        control_type='clf_dro_cbf',
        planner_type='bubble',
        use_dynamic_obstacles=False,
        max_duration=30.0
    )
    
    # Print results
    print("\nExecution completed!")
    print(f"Success: {result['success']}")
    print(f"Timeout: {result['timeout']}")
    print(f"Total time: {result['total_time']:.2f}s")
    print(f"Trajectory length: {len(result['tracked_configs'])}") 