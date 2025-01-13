import pybullet as p
import torch
import numpy as np
from xarm_planning import XArmSDFVisualizer
from typing import List, Tuple
#from control.clf_cbf_qp import ClfCbfQpController
from control.clf_cbf_qp import ClfCbfQpController
from control.clf_dro_cbf import ClfCbfDrccpController


from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
import time
import os
import pickle
from control.reference_governor_bezier import BezierReferenceGovernor

class XArmController:
    def __init__(self, planner: XArmSDFVisualizer, control_type='pd'):
        self.planner = planner
        self.env = planner.env
        self.robot_id = self.env.robot_id
        self.device = planner.device
        self.control_type = control_type
        
        # Control parameters
        self.max_velocity = 2.0  # Matches test case

        self.num_samples = 10
        
        # Initialize controllers
        self.pd_controller = PDController(kp=0.8, kd=0.1, max_velocity=self.max_velocity)
        self.clf_cbf_controller = ClfCbfQpController(
            p1=1e0,    # Control effort penalty
            p2=1e3,    # CLF slack variable penalty
            clf_rate=1.0,
            cbf_rate=0.5,
            safety_margin=0.2
        )
        # Add DR-CLF-CBF controller
        self.clf_dro_cbf_controller = ClfCbfDrccpController(
            p1=1e0,    # Control effort penalty
            p2=1e2,    # CLF slack variable penalty
            clf_rate=1.0,
            cbf_rate=0.5,
            wasserstein_r=0.01,
            epsilon=0.1,
            num_samples=self.num_samples  # Number of CBF samples to use
        )

    def compute_control(self, current_joints: torch.Tensor, 
                       target_joints: torch.Tensor,
                       current_vel: torch.Tensor) -> torch.Tensor:
        """Compute control based on selected controller type"""
        if self.control_type == 'pd':
            velocity_cmd = self.pd_controller.compute_control(
                current_joints,
                target_joints,
                current_vel
            )
            print('PD velocity_cmd', velocity_cmd)
            
        elif self.control_type == 'clf_cbf':
            # Ensure we need gradients
            current_joints.requires_grad_(True)
            
            # Get SDF values without computing gradients in query_sdf
            h_values = self.planner.robot_cdf.query_cdf(
                points=self.planner.points_robot.unsqueeze(0),
                joint_angles=current_joints.unsqueeze(0),
                return_gradients=False  # Don't compute gradients here
            )
            
            # Get minimum SDF/CDF value
            h = h_values.min() - 0.35
            
            # Compute gradient through backprop
            h.backward()
            dh_dtheta = current_joints.grad.clone()
            current_joints.grad.zero_()  # Clear gradients

            #print('cbf value', h.item() - 0.2)
            #print('cbf gradient', dh_dtheta)
            
            # Generate control input using CLF-CBF-QP
            velocity_cmd = self.clf_cbf_controller.generate_controller(
                current_joints.detach().cpu().numpy(),
                target_joints.cpu().numpy(),
                h.item(),
                dh_dtheta.cpu().numpy()
            )
            velocity_cmd = torch.tensor(velocity_cmd, device=self.device)
        
        elif self.control_type == 'clf_dro_cbf':
            # Ensure we need gradients
            current_joints.requires_grad_(True)
            
            # Get SDF values for multiple points
            h_values = self.planner.robot_cdf.query_cdf(
                points=self.planner.points_robot.unsqueeze(0),
                joint_angles=current_joints.unsqueeze(0),
                return_gradients=False
            )

            print('cdf value:', h_values.min().item())
            
            # Get the k smallest SDF values
            k = self.num_samples  # Number of samples to use
            h_values_flat = h_values.flatten()
            top_k_values, _ = torch.topk(h_values_flat, k, largest=False)
            h_samples = top_k_values 
            
            # Initialize gradient storage
            h_grads = []
            
            # Compute gradients for each of the k smallest values
            for h_val in h_samples:
                h_val.backward(retain_graph=True)
                h_grads.append(current_joints.grad.clone().cpu().numpy())
                current_joints.grad.zero_()
            
            # Convert to numpy arrays
            h_samples_np = h_samples.detach().cpu().numpy()
            h_grads_np = np.stack(h_grads)
            dh_dt_samples = np.zeros(k)  # Assuming static environment
            
            # Generate control input using DR-CLF-CBF-QP
            velocity_cmd = self.clf_dro_cbf_controller.generate_controller(
                current_joints.detach().cpu().numpy(),
                target_joints.cpu().numpy(),
                h_samples_np,
                h_grads_np,
                dh_dt_samples
            )
            velocity_cmd = torch.tensor(velocity_cmd, device=self.device)
        
        # Clip velocities
        velocity_cmd = torch.clamp(velocity_cmd, -self.max_velocity, self.max_velocity)
        
        return velocity_cmd

    def downsample_trajectory(self, trajectory: List[np.ndarray], num_points: int = 10) -> np.ndarray:
        """Downsample trajectory to specified number of points"""
        if len(trajectory) <= num_points:
            return np.array(trajectory)
        
        # Create indices for evenly spaced samples
        indices = np.linspace(10, len(trajectory)-1, num_points, dtype=int)
        return np.array(trajectory)[indices]

    def execute_trajectory(self, trajectory_data: dict, dt: float = 0.02, use_bezier: bool = True) -> Tuple[List[float], List[float]]:
        """
        Execute trajectory using velocity control.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
                - waypoints: Discrete waypoints
                - bezier_curves: Bezier curve segments (optional)
                - times: Time parameterization
            dt: Time step for control
            use_bezier: Whether to use Bezier curves (True) or discrete waypoints (False)
        """
        goal_distances = []
        sdf_distances = []
        
        initial_state = self.planner.get_current_joint_states().cpu().numpy()
        
        # Initialize appropriate governor based on mode
        if use_bezier and 'bezier_curves' in trajectory_data:
            print("Using Bezier-based reference governor")
            governor = BezierReferenceGovernor(
                initial_state=initial_state,
                trajectory_data=trajectory_data,
                dt=dt
            )
        else:
            print("Using discrete waypoint-based reference governor")
            waypoints = (trajectory_data['waypoints'] if isinstance(trajectory_data, dict) 
                        else trajectory_data)
            governor = ReferenceGovernor(
                initial_state=initial_state,
                path_configs=waypoints,
                dt=dt
            )

        while True:
            # Get current joint states and velocities
            current_joints = self.planner.get_current_joint_states()
            current_velocities = torch.tensor([
                p.getJointState(self.robot_id, i+1)[1]
                for i in range(6)
            ], device=self.device)

            # Get reference based on governor type
            if use_bezier and 'bezier_curves' in trajectory_data:
                reference_joints, s, reference_vel = governor.update(current_joints.cpu().numpy())
            else:
                reference_joints, s = governor.update(current_joints.cpu().numpy())
            
            target_joints = torch.tensor(reference_joints, device=self.device)
            
            # Compute and apply control
            velocity_cmd = self.compute_control(
                current_joints, 
                target_joints,
                current_velocities
            )

            # Apply velocity commands
            for i in range(6):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i+1,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=velocity_cmd[i].item(),
                    force=100
                )

            # Step simulation
            p.stepSimulation()
            time.sleep(dt)

            # Record metrics
            current_ee_pos = self.planner.get_ee_position(current_joints)
            goal_dist = torch.norm(self.planner.goal - current_ee_pos.squeeze())
            goal_distances.append(goal_dist.item())

            # Break if we're close to the end of the trajectory
            if s > 0.99 and goal_dist < 0.02:
                break

        return goal_distances, sdf_distances

if __name__ == "__main__":
    # Example usage
    goal_pos = torch.tensor([0.7, 0.2, 0.6], device='cuda')
    planner_type = 'bubble_cdf'  # or other types
    
    # Define cache file path
    cache_dir = 'trajectory_cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, 
        f'traj_{planner_type}_goal_{goal_pos[0]:.2f}_{goal_pos[1]:.2f}_{goal_pos[2]:.2f}.pkl'
    )
    
    # Try to load from cache first
    trajectory_whole = None
    if os.path.exists(cache_file):
        print(f"Loading trajectory from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            trajectory_whole = pickle.load(f)
    
    # If not in cache, plan new trajectory
    if trajectory_whole is None:
        print("Planning new trajectory...")
        planner = XArmSDFVisualizer(goal_pos, use_gui=True, planner_type=planner_type)
        trajectory_whole = planner.run_demo(execute_trajectory=False)
        
        # Save to cache if planning successful
        if trajectory_whole is not None:
            print(f"Saving trajectory to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(trajectory_whole, f)
        
        # Disconnect from PyBullet before creating new instance
        p.disconnect()
    
    if trajectory_whole is not None:
        # Initialize controller with PD control
        planner = XArmSDFVisualizer(goal_pos, use_gui=True, planner_type=planner_type)
        controller = XArmController(planner, control_type='clf_dro_cbf')     # baselines: 'pd', 'clf_cbf' 
        
        if planner_type == 'bubble_cdf':
            # For bubble_cdf planner, trajectory_whole is a dictionary
            print(f"Original trajectory length: {len(trajectory_whole['waypoints'])}")
            distances = controller.execute_trajectory(trajectory_whole, use_bezier=True)
        else:
            # For other planners, trajectory_whole is directly the waypoints
            print(f"Original trajectory length: {len(trajectory_whole)}")
            distances = controller.execute_trajectory(trajectory_whole)
    else:
        print("Failed to generate trajectory!") 