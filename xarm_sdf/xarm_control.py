import pybullet as p
import torch
import numpy as np
from xarm_planning import XArmSDFVisualizer
from typing import List, Tuple
from control.clf_cbf_qp import ClfCbfQpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
import time
import os
import pickle

class XArmController:
    def __init__(self, planner: XArmSDFVisualizer, control_type='pd'):
        self.planner = planner
        self.env = planner.env
        self.robot_id = self.env.robot_id
        self.device = planner.device
        self.control_type = control_type
        
        # Control parameters
        self.max_velocity = 2.0  # Matches test case
        
        # Initialize controllers
        self.pd_controller = PDController(kp=0.8, kd=0.1, max_velocity=self.max_velocity)
        self.clf_cbf_controller = ClfCbfQpController(
            p1=1e0,    # Control effort penalty
            p2=1e3,    # CLF slack variable penalty
            clf_rate=1.0,
            cbf_rate=0.5,
            safety_margin=0.2
        )

    def compute_control(self, current_pos: torch.Tensor, 
                       target_pos: torch.Tensor,
                       current_vel: torch.Tensor) -> torch.Tensor:
        """Compute control based on selected controller type"""
        if self.control_type == 'pd':
            velocity_cmd = self.pd_controller.compute_control(
                current_pos,
                target_pos,
                current_vel
            )
            print('PD velocity_cmd', velocity_cmd)
            
        else:  # clf_cbf
            if self.control_type == 'clf_cbf':
                # Ensure we need gradients
                current_pos.requires_grad_(True)
                
                # Get SDF values without computing gradients in query_sdf
                sdf_values = self.planner.robot_sdf.query_sdf(
                    points=self.planner.points_robot.unsqueeze(0),
                    joint_angles=current_pos.unsqueeze(0),
                    return_gradients=False  # Don't compute gradients here
                )
                
                # Get minimum SDF value
                h = sdf_values.min()
                
                # Compute gradient through backprop
                h.backward()
                dh_dtheta = current_pos.grad.clone()
                current_pos.grad.zero_()  # Clear gradients

                print('cbf value', h.item())
                print('cbf gradient', dh_dtheta)
                
                # Generate control input using CLF-CBF-QP
                velocity_cmd = self.clf_cbf_controller.generate_controller(
                    current_pos.detach().cpu().numpy(),
                    target_pos.cpu().numpy(),
                    h.item(),
                    dh_dtheta.cpu().numpy()
                )
                velocity_cmd = torch.tensor(velocity_cmd, device=self.device)
            else:
                # Convert to numpy for the controller
                current_config = current_pos.cpu().numpy()
                reference_config = target_pos.cpu().numpy()
                
                # Get CBF value and gradient from the SDF
                start_sdf = time.time()
                sdf_values, gradients = self.planner.robot_sdf.query_sdf(
                    points=self.planner.points_robot.unsqueeze(0),
                    joint_angles=current_pos.unsqueeze(0),
                    return_gradients=True,
                    gradient_method='analytic'
                )
                
                # Get minimum SDF value as CBF value
                h = sdf_values.min().item()
                min_idx = sdf_values.argmin()
                dh_dtheta = gradients[0, min_idx].detach().cpu().numpy()
                
                
                # Generate control input using CLF-CBF-QP
                start_qp = time.time()
                velocity_cmd = self.clf_cbf_controller.generate_controller(
                    current_config,
                    reference_config,
                    h,
                    dh_dtheta
                )
                
                
                #print('Raw CLF-CBF velocity_cmd', velocity_cmd)
                # velocity_cmd = velocity_cmd * 100.0  # Scale up by 100x
                #print('Scaled CLF-CBF velocity_cmd', velocity_cmd)
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

    def execute_trajectory(self, trajectory: List[np.ndarray], dt: float = 0.02) -> Tuple[List[float], List[float]]:
        """Execute trajectory using velocity control"""
        goal_distances = []
        sdf_distances = []
        
        # Print original trajectory info
        # print("\nOriginal Trajectory:")
        # print(f"First config: {trajectory[0]}")
        # print(f"Last config: {trajectory[-1]}")
        
        # Downsample trajectory
        downsampled_traj = self.downsample_trajectory(trajectory, num_points=100)
        print(f"\nDownsampled trajectory from {len(trajectory)} to {len(downsampled_traj)} points")
        
        # Initialize reference governor
        initial_state = self.planner.get_current_joint_states().cpu().numpy()
        print(f"\nInitial robot state: {initial_state}")
        
        governor = ReferenceGovernor(
            initial_state=initial_state,
            path_configs=trajectory,
            dt=dt
        )

        while True:
            # Get current joint states
            current_joints = self.planner.get_current_joint_states()
            
            # Get reference from governor
            reference_joints, s = governor.update(current_joints.cpu().numpy())
            target_joints = torch.tensor(reference_joints, device=self.device)
            
            # Get current joint velocities
            current_velocities = torch.tensor([
                p.getJointState(self.robot_id, i+1)[1]
                for i in range(6)
            ], device=self.device)

            start_time = time.time()

            # Compute control
            velocity_cmd = self.compute_control(
                current_joints, 
                target_joints,
                current_velocities
            )
            compute_time = time.time() - start_time
            print(f"Control computation time: {compute_time:.4f} seconds")

            # Apply velocity commands (matching test case)
            for i in range(6):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i+1,  # Note: using i+1 for joint index
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=velocity_cmd[i].item(),
                    force=100  # Consistent with test case
                )

            # Step simulation
            p.stepSimulation()
            time.sleep(dt)

            # Record metrics
            current_ee_pos = self.planner.get_ee_position(current_joints)
            goal_dist = torch.norm(self.planner.goal - current_ee_pos.squeeze())
            goal_distances.append(goal_dist.item())

            # Print progress
            #print(f"Progress: {s:.2f}, Goal distance: {goal_dist:.4f}")

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
        controller = XArmController(planner, control_type='clf_cbf')
        
        if planner_type == 'bubble_cdf':
            # For bubble_cdf planner, trajectory_whole is a dictionary
            print(f"Original trajectory length: {len(trajectory_whole['waypoints'])}")
            distances = controller.execute_trajectory(trajectory_whole['waypoints'])
        else:
            # For other planners, trajectory_whole is directly the waypoints
            print(f"Original trajectory length: {len(trajectory_whole)}")
            distances = controller.execute_trajectory(trajectory_whole)
    else:
        print("Failed to generate trajectory!") 