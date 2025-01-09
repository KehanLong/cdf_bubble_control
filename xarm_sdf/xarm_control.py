import pybullet as p
import torch
import numpy as np
from xarm_planning import XArmSDFVisualizer
from typing import List, Tuple
from control.clf_cbf_qp import ClfCbfQpController
from control.reference_governor import ReferenceGovernor
from control.pd_control import PDController
import time

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
            cbf_rate=1.0,
            safety_margin=0.1
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
            # Convert to numpy for the controller
            current_config = current_pos.cpu().numpy()
            reference_config = target_pos.cpu().numpy()
            
            # Get CBF value and gradient from the SDF
            sdf_values, gradients = self.planner.robot_sdf.query_sdf(
                points=self.planner.points_robot.unsqueeze(0),
                joint_angles=current_pos.unsqueeze(0),
                return_gradients=True,
                gradient_method='analytic'
            )
            
            # Get minimum SDF value as CBF value
            h = sdf_values.min().item()
            min_idx = sdf_values.argmin()
            dh_dtheta = gradients[0, min_idx].cpu().numpy()
            
            # Generate control input using CLF-CBF-QP
            velocity_cmd = self.clf_cbf_controller.generate_controller(
                current_config,
                reference_config,
                h,
                dh_dtheta
            )
            
            print('Raw CLF-CBF velocity_cmd', velocity_cmd)
            velocity_cmd = velocity_cmd * 100.0  # Scale up by 100x
            print('Scaled CLF-CBF velocity_cmd', velocity_cmd)
            velocity_cmd = torch.tensor(velocity_cmd, device=self.device)
        
        # Clip velocities
        velocity_cmd = torch.clamp(velocity_cmd, -self.max_velocity, self.max_velocity)
        print('Final velocity_cmd', velocity_cmd)
        
        return velocity_cmd

    def downsample_trajectory(self, trajectory: List[np.ndarray], num_points: int = 10) -> np.ndarray:
        """Downsample trajectory to specified number of points"""
        if len(trajectory) <= num_points:
            return np.array(trajectory)
        
        # Create indices for evenly spaced samples
        indices = np.linspace(10, len(trajectory)-1, num_points, dtype=int)
        return np.array(trajectory)[indices]

    def execute_trajectory(self, trajectory: List[np.ndarray], dt: float = 0.01) -> Tuple[List[float], List[float]]:
        """Execute trajectory using velocity control"""
        goal_distances = []
        sdf_distances = []
        
        # Print original trajectory info
        print("\nOriginal Trajectory:")
        print(f"First config: {trajectory[0]}")
        print(f"Last config: {trajectory[-1]}")
        
        # Downsample trajectory
        downsampled_traj = self.downsample_trajectory(trajectory, num_points=30)
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
            
            # Get current joint velocities (matching test case indexing)
            current_velocities = torch.tensor([
                p.getJointState(self.robot_id, i+1)[1]  # Note: using i+1 for joint index
                for i in range(6)
            ], device=self.device)

            # Compute control
            velocity_cmd = self.compute_control(
                current_joints, 
                target_joints,
                current_velocities
            )

            print(f"\nCurrent joints: {current_joints}")
            print(f"Target joints: {target_joints}")
            print(f"Velocity command: {velocity_cmd}")

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
            print(f"Progress: {s:.2f}, Goal distance: {goal_dist:.4f}")

            # Break if we're close to the end of the trajectory
            if s > 0.99 and goal_dist < 0.02:
                break

        return goal_distances, sdf_distances

if __name__ == "__main__":
    # Example usage
    goal_pos = torch.tensor([0.7, 0.2, 0.6], device='cuda')
    planner = XArmSDFVisualizer(goal_pos, use_gui=True, planner_type='bubble_cdf')
    
    # Get trajectory without executing it
    trajectory = planner.run_demo(execute_trajectory=False)
    
    if trajectory is not None:
        # Initialize controller with PD control
        controller = XArmController(planner, control_type='clf_cbf')  # or 'clf_cbf'
        
        print(f"Original trajectory length: {len(trajectory)}")
        # Execute trajectory with velocity control
        distances = controller.execute_trajectory(trajectory)
    else:
        print("Failed to generate trajectory!") 