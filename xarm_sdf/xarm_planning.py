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
from planner.mppi_functional import setup_mppi_controller
from models.xarm_model import XArmFK
from planner.rrt_functional import RRTPlanner, RRTConnectPlanner
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
    def __init__(self, ee_goal, use_gui=True, initial_horizon=8, planner_type='bubble_cdf', seed=5):
        """
        Initialize XArmSDFVisualizer
        
        Args:
            ee_goal: End-effector goal position
            use_gui: Whether to use GUI visualization
            initial_horizon: Initial horizon for MPPI
            planner_type: Type of planner to use ('bubble_cdf', 'rrt', 'rrt_connect', 'mppi')
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
        self.env = XArmEnvironment(gui=use_gui, add_dynamic_obstacles=False)
        self.physics_client = self.env.client
        self.robot_id = self.env.robot_id
        self.use_gui = use_gui

        # Robot base transform
        self.base_pos = torch.tensor(self.env.robot_base_pos, device='cuda')


        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize robot FK and SDF model
        self.robot_fk = XArmFK(device=self.device)
        self.robot_sdf = RobotSDF(device=self.device)
        self.robot_cdf = RobotCDF(device=self.device)
        # Store goal configuration (in task space)
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
        elif planner_type == 'rrt':
            # Initialize OMPL RRT planner with seed
            joint_limits = (
                self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                self.robot_fk.joint_limits[:, 1].cpu().numpy()
            )
            self.rrt_planner = OMPLRRTPlanner(
                robot_sdf=self.robot_sdf,
                robot_fk=self.robot_fk,
                joint_limits=joint_limits,
                device=self.device,
                seed=seed  # Pass seed to planner
            )
        elif planner_type == 'rrt_connect':
            # Initialize RRT-Connect planner
            joint_limits = (
                self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                self.robot_fk.joint_limits[:, 1].cpu().numpy()
            )
            self.rrt_planner = RRTConnectPlanner(
                robot_sdf=self.robot_sdf,
                robot_fk=self.robot_fk,
                joint_limits=joint_limits,
                step_size=0.2,  # Same adaptive step size as RRT
                max_nodes=1e6,
                batch_size=1000,
                goal_bias=0.1,  # Less important for RRT-Connect
                device=self.device
            )
        elif planner_type == 'bubble_cdf':
            # Initialize Bubble Planner with seed
            self.bubble_planner = BubblePlanner(
                robot_cdf=self.robot_cdf,
                joint_limits=(
                    self.robot_fk.joint_limits[:, 0].cpu().numpy(),
                    self.robot_fk.joint_limits[:, 1].cpu().numpy()
                ),
                device=self.device,
                seed=seed  # Pass seed to planner
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        
        # Create goal marker
        self.create_goal_marker(self.goal)
        
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

    def create_goal_marker(self, goal_config):
        """Create a visible marker for the goal position"""
        self.goal_marker = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0, 1, 0, 0.7]  # Green, semi-transparent
        )
        
        self.goal_visual = p.createMultiBody(
            baseVisualShapeIndex=self.goal_marker,
            basePosition=goal_config.cpu().numpy(),
            baseOrientation=[0, 0, 0, 1]
        )

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

    def _find_goal_configuration(self, goal_pos: torch.Tensor, n_samples: int = 1e6, threshold: float = 0.05, seed: int = None) -> Optional[np.ndarray]:
        """
        Find valid goal configurations for a given goal position using random sampling
        
        Args:
            goal_pos: Target position in task space
            n_samples: Number of samples to try
            threshold: Distance threshold for considering a configuration valid
            seed: Random seed for reproducibility
        """
        from utils.find_goal_pose import find_goal_configuration
        
        print(f"\nSearching for goal configurations:")
        print(f"Target position: {goal_pos.cpu().numpy()}")
        print(f"Using random seed: {seed}")
        
        valid_solutions = find_goal_configuration(
            goal_pos=goal_pos,
            points_robot=self.points_robot,
            robot_fk=self.robot_fk,
            robot_sdf=self.robot_sdf,
            robot_cdf=self.robot_cdf,
            n_samples=int(n_samples),
            threshold=threshold,
            device=self.device,
            max_solutions=1,
            seed=seed  # Pass the seed to find_goal_configuration
        )
        
        if not valid_solutions:
            print("Failed to find any valid goal configurations")
            return None
        
        # Get current configuration
        current_config = self.get_current_joint_states().cpu().numpy()
        
        # Find configuration closest to current configuration in joint space
        closest_idx = 0
        min_joint_dist = float('inf')
        
        for i, (config, task_dist, min_sdf, min_cdf) in enumerate(valid_solutions):
            joint_dist = np.linalg.norm(config - current_config)
            if joint_dist < min_joint_dist:
                min_joint_dist = joint_dist
                closest_idx = i
        
        print(f"\nSelected configuration {closest_idx + 1}/{len(valid_solutions)}:")
        print(f"Joint space distance: {min_joint_dist:.4f}")
        config, task_dist, min_sdf, min_cdf = valid_solutions[closest_idx]
        print(f"Task space distance: {task_dist:.4f} meters")
        print(f"Minimum SDF value: {min_sdf:.4f}")
        print(f"Minimum CDF value: {min_cdf:.4f}")
        
        # For visualization, show each configuration briefly
        original_config = self.get_current_joint_states()
        
        print("\nVisualizing found configurations...")
        for i, (config, task_dist, min_sdf, min_cdf) in enumerate(valid_solutions):
            print(f"\nConfiguration {i+1}/{len(valid_solutions)}:")
            print(f"Task space distance: {task_dist:.4f} meters")
            print(f"Joint space distance: {np.linalg.norm(config - current_config):.4f}")
            print(f"Minimum SDF value: {min_sdf:.4f}")
            print(f"Minimum CDF value: {min_cdf:.4f}")
            self.set_robot_configuration(config)
            time.sleep(4)  # Show each config for 1 second
        
        # Reset to original configuration
        print("\nResetting to original configuration...")
        self.set_robot_configuration(original_config)
        
        # Return the configuration closest to current configuration in joint space
        return valid_solutions[closest_idx][0]
    
    def _capture_frame(self, step: int, time_steps: int, width: int, height: int) -> np.ndarray:
        """Capture a frame using a rotating camera"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 1.5],
            distance=2.5,
            yaw=0,  # Rotating camera, -(step / time_steps) * 60 for rotate to left
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
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

    def run_demo(self, duration=5.0, fps=20, execute_trajectory=True) -> Union[PlanningMetrics, Tuple[List, List]]:
        print(f"Starting {self.planner_type.upper()} demo...")
        
        # Initialize lists to store distances and video frames
        goal_distances = []
        sdf_distances = []
        frames = []
        trajectory = None
        
        # Setup parameters
        width = 1920
        height = 1080
        time_steps = int(duration * fps)
        dt = 1.0 / fps
        step = 0
        
        if self.planner_type in ['rrt', 'rrt_connect']:
            # RRT/RRT-Connect planning code
            goal_config = self._find_goal_configuration(self.goal - self.base_pos)
            if goal_config is None:
                print("Failed to find valid goal configuration!")
                return None if not execute_trajectory else (goal_distances, sdf_distances)
            
            # Get current state
            current_state = self.get_current_joint_states()
            
            # Plan with OMPL RRT using only static points
            result = self.rrt_planner.plan(
                start_config=current_state.cpu().numpy(),
                goal_config=goal_config,
                obstacle_points=self.points_robot,  # Using only static points for planning
                max_time=30.0
            )
            
            if result['metrics'].success == False:
                print("RRT planning failed!")
                return None if not execute_trajectory else (goal_distances, sdf_distances)
            
            # Print planning metrics
            metrics = result['metrics']
            print("\nPlanning Statistics:")
            print(f"Planning time: {metrics.planning_time:.3f} seconds")
            print(f"Path length: {metrics.path_length:.3f}")
            print(f"Number of collision checks: {metrics.num_collision_checks}")
            print(f"Number of samples: {metrics.num_samples}")
            print("---")
                
            trajectory = result['waypoints']
            print(f"RRT path found with {len(trajectory)} waypoints")
            
            if not execute_trajectory:
                return result['metrics']
            
            # Execute trajectory
            print(f"\nExecuting {len(trajectory)} waypoints...")
            for traj_idx in range(len(trajectory)):
                # Update robot state
                next_config = trajectory[traj_idx]
                next_config_tensor = torch.tensor(next_config, device=self.device, dtype=torch.float32)
                self.set_robot_configuration(next_config_tensor)
                p.stepSimulation()
                
                # Record metrics
                current_ee_pos = self.get_ee_position(next_config_tensor)
                goal_dist = torch.norm(self.goal - current_ee_pos.squeeze())
                goal_distances.append(goal_dist.detach().cpu().numpy())
                
                sdf_values = self.robot_sdf.query_sdf(
                    points=self.points_robot.unsqueeze(0),
                    joint_angles=next_config_tensor.unsqueeze(0),
                    return_gradients=False
                )
                min_sdf = sdf_values.min()
                sdf_distances.append(min_sdf.detach().cpu().numpy())
                
                # Capture frame
                if self.use_gui:
                    frames.append(self._capture_frame(traj_idx, len(trajectory), width, height))
                    time.sleep(dt)
                
                if traj_idx % 10 == 0:
                    print(f"Waypoint {traj_idx}/{len(trajectory)}")
                    print(f"Distance to goal: {goal_dist.item():.4f}")
                    print(f"Minimum SDF value: {min_sdf.item():.4f}")
                    print("---")
                
                step += 1
                torch.cuda.empty_cache()
                
        elif self.planner_type == 'bubble_cdf':
            # Get start configuration
            current_state = self.get_current_joint_states().cpu().numpy()
            
            # Convert goal from task space to joint space
            goal_config = self._find_goal_configuration(self.goal - self.base_pos)
            if goal_config is None:
                print("Failed to find valid goal configuration!")
                return None if not execute_trajectory else (goal_distances, sdf_distances)
            
            # Plan path using bubble planner
            try:
                trajectory_whole = self.bubble_planner.plan(
                    start_config=current_state,
                    goal_config=goal_config,
                    obstacle_points=self.points_robot  # Using only static points for planning
                )

                if trajectory_whole is None:
                    print("Bubble planning failed!")
                    return None if not execute_trajectory else (goal_distances, sdf_distances)
                
                # Print planning metrics
                metrics = trajectory_whole['metrics']
                print("\nPlanning Statistics:")
                print(f"Planning time: {metrics.planning_time:.3f} seconds")
                print(f"Path length: {metrics.path_length:.3f}")
                print(f"Number of collision checks: {metrics.num_collision_checks}")
                print(f"Number of samples: {metrics.num_samples}")
                print("---")
                
                trajectory = trajectory_whole['waypoints']
                print(f"Bubble planner found path with {len(trajectory)} waypoints")
                
                if not execute_trajectory:
                    return trajectory_whole
                    
                # Execute trajectory
                traj_idx = 0
                print(f"Starting execution of {len(trajectory)} waypoints...")
                
                while traj_idx < len(trajectory):  # Remove step < time_steps condition
                    # Update robot state with explicit float32 conversion
                    next_config = trajectory[traj_idx].astype(np.float32)
                    next_config_tensor = torch.tensor(next_config, device=self.device, dtype=torch.float32)
                    self.set_robot_configuration(next_config_tensor)
                    p.stepSimulation()
                    
                    # Record metrics
                    current_ee_pos = self.get_ee_position(next_config_tensor)
                    goal_dist = torch.norm(self.goal - current_ee_pos.squeeze())
                    goal_distances.append(goal_dist.detach().cpu().numpy())

                    cdf_values = self.robot_cdf.query_cdf(
                        points=self.points_robot.unsqueeze(0),
                        joint_angles=next_config_tensor.unsqueeze(0),
                        return_gradients=False
                    )
                    min_cdf = cdf_values.min()
                    # print('cdf value', min_cdf)
                    
                    # Use SDF for distance checking during execution
                    sdf_values = self.robot_sdf.query_sdf(
                        points=self.points_robot.unsqueeze(0),
                        joint_angles=next_config_tensor.unsqueeze(0),
                        return_gradients=False
                    )
                    min_sdf = sdf_values.min()
                    sdf_distances.append(min_sdf.detach().cpu().numpy())
                    
                    # Capture frame
                    if self.use_gui:
                        frames.append(self._capture_frame(traj_idx, len(trajectory), width, height))
                        time.sleep(dt)
                    
                    # Print progress
                    if traj_idx % 10 == 0:
                        print(f"Waypoint {traj_idx}/{len(trajectory)}")
                        print(f"Distance to goal: {goal_dist.item():.4f}")
                        print(f"Minimum SDF value: {min_sdf.item():.4f}")
                        print("---")
                    
                    traj_idx += 1
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error during bubble planning execution: {str(e)}")
                return None if not execute_trajectory else (goal_distances, sdf_distances)
            
        else:  # MPPI
            current_state = self.get_current_joint_states()
            U = torch.zeros((self.initial_horizon, 6), device=self.device)
            trajectory = []
            
            while step < time_steps:
                # MPPI control step
                states_final, action, U = self.mppi_controller(
                    key=None,
                    U=U,
                    init_state=current_state,
                    goal=self.goal,
                    obstaclesX=self.points_robot,
                    safety_margin=0.01
                )
                trajectory.append(current_state.cpu().numpy())
                
                if not execute_trajectory:
                    continue
                    
                # Update robot state
                next_state = current_state + action.squeeze() * dt
                self.set_robot_configuration(next_state)
                p.stepSimulation()
                current_state = next_state
                
                # Record metrics
                current_ee_pos = self.get_ee_position(current_state)
                goal_dist = torch.norm(self.goal - current_ee_pos.squeeze())
                goal_distances.append(goal_dist.detach().cpu().numpy())
                
                sdf_values = self.robot_sdf.query_sdf(
                    points=self.points_robot.unsqueeze(0),
                    joint_angles=current_state.unsqueeze(0),
                    return_gradients=False
                )
                min_sdf = sdf_values.min()
                sdf_distances.append(min_sdf.detach().cpu().numpy())
                
                # Capture frame
                if self.use_gui:
                    frames.append(self._capture_frame(step, time_steps, width, height))
                    time.sleep(dt)
                
                # Print progress
                if step % 10 == 0:
                    print(f"Step {step}/{time_steps}")
                    print(f"Distance to goal: {goal_dist.item():.4f}")
                    print(f"Minimum SDF value: {min_sdf.item():.4f}")
                    print("---")
                
                step += 1
                torch.cuda.empty_cache()
            
            if not execute_trajectory:
                return trajectory
        
        # Save video and plot results
        if frames:
            print("Saving video...")
            imageio.mimsave(f'{self.planner_type}_demo.mp4', frames, fps=fps)
        
        print("Demo completed. Plotting distances...")
        plot_distances(np.array(goal_distances), np.array(sdf_distances), obst_radius=0.05, dt=dt)
        
        return goal_distances, sdf_distances

if __name__ == "__main__":
    # Example usage for comparison
    goal_pos = torch.tensor([0.2, 0.6, 0.6], device='cuda')     

    # example goal pos: [0.7, 0.2, 0.6],      

    seed = 5
    visualizer = XArmSDFVisualizer(goal_pos, use_gui=True, planner_type='bubble_cdf', seed=seed)  # baselines: 'rrt', 'rrt_connect', 'mppi'

    # single demo
    visualizer.run_demo(execute_trajectory=True)  
    