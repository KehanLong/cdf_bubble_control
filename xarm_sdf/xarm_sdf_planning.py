import pybullet as p
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import imageio
import os
from xarm_sim_env import XArmEnvironment
from robot_sdf import RobotSDF
from mppi_functional import setup_mppi_controller
from models.xarm_model import XArmFK
from rrt_functional import RRTPlanner, RRTConnectPlanner
from typing import Optional

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

class XArmSDFVisualizer:
    def __init__(self, ee_goal, use_gui=True, initial_horizon=8, planner_type='mppi'):
        # Initialize environment
        self.env = XArmEnvironment(gui=use_gui)
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
            # Initialize RRT planner with adjusted step size
            joint_limits = (
                self.robot_fk.joint_limits[:, 0].cpu().numpy(),  # lower limits
                self.robot_fk.joint_limits[:, 1].cpu().numpy()   # upper limits
            )
            self.rrt_planner = RRTPlanner(
                robot_sdf=self.robot_sdf,
                robot_fk=self.robot_fk,
                joint_limits=joint_limits,
                step_size=0.2,  # Increased from 0.1 to allow larger initial steps
                max_nodes=1e6,
                batch_size=1000,
                goal_bias=0.1,
                device=self.device
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

        # Precompute point cloud
        self.points_robot = self.env.get_point_cloud(downsample=True)
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

    def _find_goal_configuration(self, goal_pos: torch.Tensor, n_samples: int = 1e6, threshold: float = 0.05) -> Optional[np.ndarray]:
        """Find a valid goal configuration for a given goal position using random sampling"""
        print(f"\nSearching for goal configuration:")
        print(f"Target position: {goal_pos.cpu().numpy()}")
        
        # Get joint limits and ensure float32
        lower_limits = self.robot_fk.joint_limits[:, 0].cpu().numpy().astype(np.float32)
        upper_limits = self.robot_fk.joint_limits[:, 1].cpu().numpy().astype(np.float32)
        
        # Sample configurations in batches for efficiency
        batch_size = 1000
        n_batches = int(n_samples // batch_size)
        
        best_config = None
        best_dist = float('inf')
        
        for batch in range(n_batches):
            # Sample random configurations in float32
            configs = np.random.uniform(
                low=lower_limits,
                high=upper_limits,
                size=(batch_size, len(lower_limits))
            ).astype(np.float32)  # Explicitly convert to float32
            
            # Convert to tensor, ensuring float32
            configs_tensor = torch.tensor(configs, device=self.device, dtype=torch.float32)
            
            try:
                # Get end-effector positions
                ee_positions = self.robot_fk.fkine(configs_tensor)
                ee_positions = ee_positions[:, -1]  # [batch_size, 3]
                
                # Compute distances to goal
                distances = torch.norm(ee_positions - goal_pos.unsqueeze(0), dim=1)
                print(f"Distances dtype: {distances.dtype}")
                print(f"Min distance: {distances.min().item():.4f}")
                
                # Find valid configurations
                valid_indices = torch.where(distances < threshold)[0]
                
                if len(valid_indices) > 0:
                    # Get the configuration with minimum distance
                    best_idx = valid_indices[distances[valid_indices].argmin()]
                    best_config = configs[best_idx]
                    
                    print(f"Found goal configuration after {batch * batch_size + best_idx.item()} samples")
                    print(f"End-effector error: {distances[best_idx].item():.4f} meters")
                    
                    # Verify collision-free
                    try:
                        sdf_values = self.robot_sdf.query_sdf(
                            points=self.points_robot.unsqueeze(0),
                            joint_angles=torch.tensor(best_config, device=self.device, dtype=torch.float32).unsqueeze(0),
                            return_gradients=False
                        )
                        
                        if sdf_values.min() > 0.02:  # Safety margin
                            print("Configuration is collision-free")
                            
                            # Save current configuration
                            original_config = self.get_current_joint_states()
                            
                            # Set robot to goal configuration for visualization
                            print("Visualizing goal configuration for 5 seconds...")
                            self.set_robot_configuration(best_config)
                            time.sleep(5)
                            
                            # Reset to original configuration
                            print("Resetting to original configuration...")
                            self.set_robot_configuration(original_config)
                            
                            return best_config
                        else:
                            print(f"Configuration is in collision (min SDF: {sdf_values.min():.4f}), continuing search...")
                            continue
                            
                    except Exception as e:
                        print(f"Error in collision checking:")
                        print(f"Point cloud shape: {self.points_robot.shape}")
                        print(f"Error message: {str(e)}")
                        continue
                    
            except Exception as e:
                print(f"Error in batch {batch}:")
                print(f"Error message: {str(e)}")
                print(f"Configs shape: {configs_tensor.shape}")
                print(f"First config: {configs_tensor[0]}")
                continue
        
        print("Failed to find valid goal configuration")
        return None
    
    def _capture_frame(self, step: int, time_steps: int, width: int, height: int) -> np.ndarray:
        """Capture a frame using a rotating camera"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 1.5],
            distance=2.5,
            yaw=(step / time_steps) * 60,  # Rotating camera
            pitch=0,
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

    def run_demo(self, duration=5.0, fps=20):
        """Run demo with selected planner and record video"""
        print(f"Starting {self.planner_type.upper()} demo...")
        
        # Initialize lists to store distances and video frames
        goal_distances = []
        sdf_distances = []
        frames = []
        
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
                return goal_distances, sdf_distances
            
            current_state = self.get_current_joint_states()
            trajectory = self.rrt_planner.plan(
                start_config=current_state.cpu().numpy(),
                goal_config=goal_config,
                obstacle_points=self.points_robot
            )
            if not trajectory:
                print(f"{self.planner_type.upper()} planning failed!")
                return goal_distances, sdf_distances
            print(f"{self.planner_type.upper()} path found with {len(trajectory)} waypoints")
            
            # Execute trajectory
            traj_idx = 0
            while step < time_steps and traj_idx < len(trajectory):
                # Update robot state with explicit float32
                next_config = trajectory[traj_idx]
                next_config_tensor = torch.tensor(next_config, device=self.device, dtype=torch.float32)
                self.set_robot_configuration(next_config_tensor)
                p.stepSimulation()
                
                # Record metrics with float32
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
                    frames.append(self._capture_frame(step, time_steps, width, height))
                    time.sleep(dt)
                
                if step % 10 == 0:
                    print(f"Step {step}/{time_steps}")
                    print(f"Distance to goal: {goal_dist.item():.4f}")
                    print(f"Minimum SDF value: {min_sdf.item():.4f}")
                    print("---")
                
                step += 1
                traj_idx += 1
                torch.cuda.empty_cache()
                
        else:  # MPPI
            current_state = self.get_current_joint_states()
            U = torch.zeros((self.initial_horizon, 6), device=self.device)
            
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
            
        # Save video and plot results
        if frames:
            print("Saving video...")
            imageio.mimsave(f'{self.planner_type}_demo.mp4', frames, fps=fps)
        
        print("Demo completed. Plotting distances...")
        plot_distances(np.array(goal_distances), np.array(sdf_distances), obst_radius=0.05, dt=dt)
        
        return goal_distances, sdf_distances

if __name__ == "__main__":
    # Example usage
    goal_pos = torch.tensor([0.1, 0.5, 0.6], device='cuda')
    
    # Select planner type ('mppi', 'rrt', or 'rrt_connect')
    planner_type = 'rrt'  # Change this to use RRT-Connect
    
    visualizer = XArmSDFVisualizer(goal_pos, use_gui=True, planner_type=planner_type)
    distances = visualizer.run_demo()