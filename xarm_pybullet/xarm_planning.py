import pybullet as p
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import imageio
import os
from xarm_sim_env import XArmEnvironment
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF

# Add project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


from planner.mppi_functional import setup_mppi_controller
from models.xarm_model import XArmFK
from planner.rrt_ompl import OMPLRRTPlanner

from typing import Optional, Union, List, Tuple, Dict
from planner.bubble_planner import BubblePlanner
from dataclasses import dataclass
from utils.find_goal_pose import find_goal_configuration

def plot_distances(goal_distances, estimated_obstacle_distances, obst_radius, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(time_steps, goal_distances, color='red', linestyle=':', linewidth=3, label='Distance to Goal')
    
    if estimated_obstacle_distances.ndim == 1:
        plt.plot(time_steps, estimated_obstacle_distances, linewidth=3, label='Distance to Obstacle')
    else:
        if estimated_obstacle_distances.ndim == 3:
            estimated_obstacle_distances = estimated_obstacle_distances.squeeze(1)
        
        num_obstacles = estimated_obstacle_distances.shape[1]
        for i in range(num_obstacles):
            if i == 0:
                plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3, label='Distance to Obstacles')
            else:
                plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3)
    
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Distance', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

@dataclass
class PlanningMetrics:
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float

class XArmSDFVisualizer:
    def __init__(self, ee_goal, use_gui=True, initial_horizon=8, planner_type='bubble', 
                 seed=5, dynamic_obstacles=False, use_pybullet_inverse=True, early_termination=False):
        """
        Initialize XArmSDFVisualizer
        
        Args:
            ee_goal: End-effector goal position
            use_gui: Whether to use GUI visualization
            initial_horizon: Initial horizon for MPPI
            planner_type: Type of planner to use ('bubble_cdf', 'cdf_rrt', 'sdf_rrt', 'mppi')
            seed: Random seed for reproducibility
        """
        # Set random seeds first
        print(f"\nInitializing with random seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Store seed for later use in planning
        self.seed = seed
        
        # Initialize environment
        self.env = XArmEnvironment(gui=use_gui, add_dynamic_obstacles=dynamic_obstacles)
        self.physics_client = self.env.client
        self.robot_id = self.env.robot_id
        self.use_gui = use_gui

        # Robot base transform
        self.base_pos = torch.tensor(self.env.robot_base_pos, device='cuda')


        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.use_pybullet_inverse = use_pybullet_inverse  # Default to random sampling
        self.early_termination = early_termination
        
        # Initialize robot FK and SDF model
        self.robot_fk = XArmFK(device=self.device)
        self.robot_sdf = RobotSDF(device=self.device)
        self.robot_cdf = RobotCDF(device=self.device)
        # Store goal pos (in task space)
        self.goal = ee_goal + self.base_pos
        self.initial_horizon = initial_horizon
        
        self.planner_type = planner_type
        if planner_type == 'mppi':
            # Initialize MPPI controller
            self.mppi_controller = setup_mppi_controller(
                robot_sdf=self.robot_sdf,
                use_GPU=(self.device=='cuda'),
                samples=400,
                initial_horizon=self.initial_horizon
            )
        elif planner_type in ['cdf_rrt', 'sdf_rrt', 'lazy_rrt', 'rrt_connect']:
            # Initialize OMPL RRT planner with seed
            joint_limits = (
                self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                self.robot_fk.joint_limits[:, 1].cpu().numpy()
            )
            self.rrt_planner = OMPLRRTPlanner(
                robot_sdf=self.robot_sdf,
                robot_cdf=self.robot_cdf,
                robot_fk=self.robot_fk,
                joint_limits=joint_limits,
                planner_type=planner_type,  # Pass planner type to use appropriate collision checker
                check_resolution=0.002,
                device=self.device,
                seed=seed
            )
        elif planner_type in ['bubble', 'bubble_connect']:
            # Initialize Bubble Planner with seed
            self.bubble_planner = BubblePlanner(
                robot_cdf=self.robot_cdf,
                joint_limits=(
                    self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                    self.robot_fk.joint_limits[:, 1].cpu().numpy()
                ),
                device=self.device,
                max_samples=1e4,                # max number of bubbles in the graph
                seed=seed,                      # Pass seed to planner
                planner_type=planner_type,      # planner type option: 'bubble' or 'bubble_connect'
                early_termination=self.early_termination
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        
        # Create goal marker
        self.env.create_goal_marker(self.goal)
        
        # Add end-effector marker
        # self.ee_marker = p.createVisualShape(
        #     p.GEOM_SPHERE,
        #     radius=0.02,
        #     rgbaColor=[1, 0, 0, 0.7]  # Red, semi-transparent
        # )
        # self.ee_visual = p.createMultiBody(
        #     baseVisualShapeIndex=self.ee_marker,
        #     basePosition=[0, 0, 0],
        #     baseOrientation=[0, 0, 0, 1]
        # )

        # Precompute static point cloud for planning
        self.points_robot = self.env.get_static_point_cloud(downsample=True)
        self.points_robot = self.points_robot.to(dtype=torch.float32)
        while self.points_robot.dim() > 2:
            self.points_robot = self.points_robot.squeeze(0)

        # Add IK solver parameters
        # self.ik_iterations = 10000  # Default number of IK iterations
        # self.ik_threshold = 0.05  # Default threshold for IK solutions (in meters)
        # self.max_ik_solutions = 10  # Maximum number of IK solutions to find

        # Add goal configuration attributes
        self.goal_configs = None
        self._found_goal_configs = False

    def set_robot_configuration(self, joint_angles):
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        
        if len(joint_angles.shape) == 2:
            joint_angles = joint_angles[0]
        
        for i in range(6):  # xArm has 6 joints
            p.resetJointState(self.robot_id, i+1, joint_angles[i])

        # link_state = p.getLinkState(self.robot_id, 6)
        # pybullet_pos = link_state[0]

        # print('pybullet_pos', pybullet_pos)

    def get_current_joint_states(self):
        joint_states = []
        for i in range(1, 7):  # xArm has 6 joints, with 1 base joint
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])
        return torch.tensor(joint_states, device=self.device)


    def get_ee_position(self, joint_angles):
        """Get end-effector position in world frame"""
        if not isinstance(joint_angles, torch.Tensor):
            joint_angles = torch.tensor(joint_angles, device=self.device)
        if len(joint_angles.shape) == 1:
            joint_angles = joint_angles.unsqueeze(0)
            
        # Get end-effector position in robot base frame
        ee_pos_local = self.robot_fk.fkine(joint_angles)[:, -1]
        
        # Debug prints
        # print("Local EE position:", ee_pos_local)
        # print("Base position:", self.base_pos)
        # print("World EE position:", ee_pos_local + self.base_pos)
        
        # # Compare with PyBullet's FK
        # state_id = p.saveState()
        # for i in range(6):
        #     p.resetJointState(self.robot_id, i+1, joint_angles[0][i])
        # link_state = p.getLinkState(self.robot_id, 6)  # 5 is typically the end-effector link
        # pybullet_ee_pos = link_state[0]
        # p.restoreState(state_id)
        # print("PyBullet EE position:", pybullet_ee_pos)
        
        # Use the same transform as in MPPI test
        ee_pos_world = ee_pos_local + self.base_pos

        # print('ee_pos_world', ee_pos_world)
        return ee_pos_world

    def update_ee_marker(self, ee_pos):
        """Update the position of the end-effector marker"""
        if isinstance(ee_pos, torch.Tensor):
            ee_pos = ee_pos.cpu().numpy()
        p.resetBasePositionAndOrientation(
            self.ee_visual,
            ee_pos,
            [0, 0, 0, 1]
        )

    def _find_goal_configuration(self, goal_pos: torch.Tensor, n_samples: int = 1e6, 
                               threshold: float = 0.05) -> Optional[List[np.ndarray]]:
        """Find valid goal configurations for a given goal position"""
        print(f"\nSearching for goal configurations:")
        print(f"Using random seed: {self.seed}")

        # debug_config = np.array([-1.417, -0.232, -0.40, 0.0, 0.632, -1.417], dtype=np.float32)
        # return [debug_config]
        
        if self.use_pybullet_inverse:
            # Use environment's IK solver

            
            # Add IK parameters
            ik_max_iterations = 5000
            ik_threshold = threshold
            ik_max_solutions = 5

            self.env.set_ik_parameters(ik_max_iterations, ik_threshold, ik_max_solutions)
            solutions = self.env.find_ik_solutions(
                target_pos=goal_pos,  # Already in robot base frame
                visualize=True,

                pause_time=1.0,
                seed=self.seed
            )
            
            if not solutions:
                print("Failed to find any valid goal configurations")
                return None
            
            # Convert tuples to numpy arrays
            goal_configs = [np.array(solution.joint_angles, dtype=np.float32) for solution in solutions]
            
            print(f"\nFound {len(goal_configs)} valid goal configurations")
            for i, solution in enumerate(solutions):
                print(f"\nSolution {i+1}:")
                print(f"Task distance: {solution.task_dist:.4f}")
                print(f"Min SDF: {solution.min_sdf:.4f}")
                print(f"Joint angles: {goal_configs[i]}")
            
            return goal_configs
        else:
            # Use existing random sampling method
            solutions = find_goal_configuration(
                goal_pos=goal_pos,
                points_robot=self.points_robot,
                robot_fk=self.robot_fk,
                robot_sdf=self.robot_sdf,
                robot_cdf=self.robot_cdf,
                n_samples=int(1e6),
                threshold=threshold,
                device=self.device,
                max_solutions=5,
                seed=self.seed
            )
            if not solutions:
                return None
            
            # Extract just the configurations from the solution tuples
            goal_configs = [config.astype(np.float32) for config, _, _, _ in solutions]
            # print('Found goal configurations:', goal_configs)
            return goal_configs
    
    def _capture_frame(self, step: int, time_steps: int, width: int, height: int) -> np.ndarray:
        """Capture a frame using a rotating camera"""

        # env1 camera parameters
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 1.5],
            distance=1.7,
            yaw=55,  # Rotating camera, -(step / time_steps) * 60 for rotate to left
            pitch=-10,
            roll=0,
            upAxisIndex=2
        )

        # env 2 camera parameters
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 1.5],
            distance=2.1,
            yaw=-40,  # Rotating camera, -(step / time_steps) * 60 for rotate to left
            pitch=-15,
            roll=0,
            upAxisIndex=2
        )


        #         camera_params = {
        #     'target': [0.0, 0.0, 1.5],
        #     'distance': 2.1,
        #     'yaw': -55,
        #     'pitch': -15,
        #     'roll': 0
        # }
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0
        )
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        return np.array(rgb)[:, :, :3]

    def run_demo(self, fps=20, execute_trajectory=True, save_snapshots=False) -> Union[PlanningMetrics, Tuple[List, List]]:
        try:
            print(f"Starting {self.planner_type.upper()} demo...")
            
            # Initialize lists to store metrics
            goal_distances = []
            sdf_distances = []
            trajectory = None
            dt = 1.0 / fps
            
            if self.planner_type in ['sdf_rrt', 'cdf_rrt', 'lazy_rrt', 'rrt_connect']:
                # RRT/RRT-Connect planning code
                if not self._found_goal_configs:
                    goal_config = self._find_goal_configuration(self.goal - self.base_pos)
                    if goal_config is None:
                        print("Failed to find valid goal configuration!")
                        return None if not execute_trajectory else (goal_distances, sdf_distances)
                    self.goal_configs = goal_config
                else:
                    goal_config = self.goal_configs

                # Get current state and plan
                current_state = self.get_current_joint_states()
                result = self.rrt_planner.plan(
                    start_config=current_state.cpu().numpy(),
                    goal_configs=goal_config,
                    obstacle_points=self.points_robot,
                    max_time=100.0,
                    early_termination=self.early_termination
                )
                
                if not result['metrics'].success:
                    print("RRT planning failed!")
                    return None if not execute_trajectory else (goal_distances, sdf_distances)
                
                trajectory = result['waypoints']
                print(f"RRT path found with {len(trajectory)} waypoints")
                
                if not execute_trajectory:
                    return result
                
            elif self.planner_type in ['bubble', 'bubble_connect']:
                # Bubble planning code
                current_state = self.get_current_joint_states().cpu().numpy()
                
                if not self._found_goal_configs:
                    goal_config = self._find_goal_configuration(self.goal - self.base_pos)
                    if goal_config is None:
                        print("Failed to find valid goal configuration!")
                        return None if not execute_trajectory else (goal_distances, sdf_distances)
                    self.goal_configs = goal_config
                else:
                    goal_config = self.goal_configs

                try:
                    trajectory_result = self.bubble_planner.plan(
                        start_config=current_state,
                        goal_configs=goal_config,
                        obstacle_points=self.points_robot
                    )
                    
                    if trajectory_result is None:
                        print("Bubble planning failed!")
                        return None if not execute_trajectory else (goal_distances, sdf_distances)
                    
                    trajectory = trajectory_result['waypoints']
                    print(f"Bubble planner found path with {len(trajectory)} waypoints")
                    
                    if not execute_trajectory:
                        return trajectory_result
                    
                except Exception as e:
                    print(f"Error during bubble planning: {str(e)}")
                    return None if not execute_trajectory else (goal_distances, sdf_distances)
            
            # Execute trajectory and record metrics
            if trajectory is not None and execute_trajectory:
                print(f"\nExecuting {len(trajectory)} waypoints...")
                
                for traj_idx in range(len(trajectory)):
                    # Update robot state
                    next_config = trajectory[traj_idx]
                    if isinstance(next_config, np.ndarray):
                        next_config = torch.tensor(next_config, device=self.device, dtype=torch.float32)
                    self.set_robot_configuration(next_config)
                    p.stepSimulation()
                    
                    # Record metrics
                    current_ee_pos = self.get_ee_position(next_config)
                    goal_dist = torch.norm(self.goal - current_ee_pos.squeeze())
                    goal_distances.append(goal_dist.detach().cpu().numpy())
                    
                    sdf_values = self.robot_sdf.query_sdf(
                        points=self.points_robot.unsqueeze(0),
                        joint_angles=next_config.unsqueeze(0),
                        return_gradients=False
                    )
                    min_sdf = sdf_values.min()
                    sdf_distances.append(min_sdf.detach().cpu().numpy())
                    
                    if traj_idx % 10 == 0:
                        print(f"Waypoint {traj_idx}/{len(trajectory)}")
                        print(f"Distance to goal: {goal_dist.item():.4f}")
                        print(f"Minimum SDF value: {min_sdf.item():.4f}")
                        print("---")
                    
                    if self.use_gui:
                        time.sleep(dt)
                    
                    torch.cuda.empty_cache()
                
                # Pause at the final configuration for 5 seconds
                if self.use_gui:
                    print("\nReached goal configuration. Pausing for 5 seconds...")
                    time.sleep(5.0)
            
            # Save snapshots, video, and plot results
            if save_snapshots and trajectory is not None:
                print("\nRecording trajectory video...")
                self.env.record_trajectory_video(
                    trajectory=trajectory,
                    fps=fps,
                    planner_type=self.planner_type
                )
                
                print("\nPlotting trajectory metrics...")
                self.env.plot_trajectory_metrics(
                    goal_distances=np.array(goal_distances),
                    sdf_distances=np.array(sdf_distances),
                    dt=dt,
                    planner_type=self.planner_type
                )
            
            return goal_distances, sdf_distances
        
        except Exception as e:
            print(f"Error during demo: {str(e)}")
            return None if not execute_trajectory else ([], [])
        
        finally:
            # Only clean up if we're executing the trajectory
            if execute_trajectory and hasattr(self, 'env'):
                print("\nClosing PyBullet environment...")
                if p.isConnected(self.physics_client):
                    p.disconnect(self.physics_client)
                self.env.close()

if __name__ == "__main__":
    # Example usage for comparison
    goal_pos = torch.tensor([0.78, 0.24, 0.37], device='cuda')     

    # env1 example goal pos: [0.7, 0.1, 0.6],  [0.2, 0.6, 0.6] , [0.22, 0.27, 0.66]   
    # env2 example goal pos: [0.78, 0.24, 0.37]


    seed = 10
    visualizer = None

    try:
        # planner_type = 'bubble', 'bubble_connect', 'sdf_rrt', 'cdf_rrt'
        visualizer = XArmSDFVisualizer(goal_pos, use_gui=True, planner_type='bubble', 
                                       seed=seed, use_pybullet_inverse=True, early_termination=True)  
        
        # visualizer.goal_configs = [np.array([-1.4397272, -1.8688867, -1.0261794,  1.4388353,  1.5402647,
        #     2.8958786], dtype=np.float32), np.array([ 1.6531591, -0.7119475, -0.9487262, -1.5808885,  1.6559765,
        #    -1.6541713], dtype=np.float32), np.array([-1.3883772, -1.7281208, -1.0457752,  1.5484365,  1.7190738,
        #     2.7941952], dtype=np.float32), np.array([-1.5003192 , -1.982543  , -0.87110025,  1.6527996 ,  1.3998513 ,
        #     2.8629906 ], dtype=np.float32), np.array([-1.5302505 ,  0.39160264,  1.570477  ,  1.411324  ,  1.4478607 ,
        #    -1.922108  ], dtype=np.float32)]

        # visualizer._found_goal_configs = True  # Set flag to skip goal config search

        # single demo
        visualizer.run_demo(
            fps=20,
            execute_trajectory=True,
            save_snapshots=True
        )
    
    finally:
        # Only try to clean up if we haven't already
        if visualizer is not None and hasattr(visualizer, 'env'):
            if p.isConnected():
                p.disconnect()  