import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import imageio
import os
import math
from panda_layer.panda_layer import PandaLayer
from bf_sdf import BPSDF
from operate_env_utils import FrankaEnvironment
from mppi_functional import setup_mppi_controller, forward_kinematics_batch, compute_robot_distances

def plot_distances(goal_distances, estimated_obstacle_distances, obst_radius, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot distance to goal as blue dotted line
    plt.plot(time_steps, goal_distances, color='red', linestyle=':', linewidth=3, label='Distance to Goal')

    # Handle both single and multiple obstacles
    if estimated_obstacle_distances.ndim == 1:
        # Single obstacle case
        plt.plot(time_steps, estimated_obstacle_distances, linewidth=3, label='Distance to Obstacle')
    else:
        # Multiple obstacles case
        # Remove the extra dimension if it exists
        if estimated_obstacle_distances.ndim == 3:
            estimated_obstacle_distances = estimated_obstacle_distances.squeeze(1)
        
        num_obstacles = estimated_obstacle_distances.shape[1]
        for i in range(num_obstacles):
            if i == 0:
                plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3, label='Distance to Obstacles')
            else:
                plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3)
    
    #plt.axhline(y=obst_radius, color='black', linestyle='--', linewidth=3, label='Safety Margin')
    
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Distance', fontsize=18)

    # Set tick label size to 16
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class PointTrajectoryManager:
    def __init__(self, num_points=10, device='cuda', bounds=None, debug_mode=False):
        self.num_points = num_points
        self.device = device
        self.debug_mode = debug_mode
        self.origin_radius = 0.3  # Radius of forbidden zone around origin
        
        # Define bounds for position [min_x, max_x, min_y, max_y, min_z, max_z]
        self.bounds = bounds if bounds is not None else torch.tensor([-1.5, 1.5, -1.5, 1.5, 0, 1.5])
        
        # Initialize positions with origin avoidance
        self.positions = self._initialize_safe_positions()
        
        # Initialize velocities
        self.velocities = torch.zeros((num_points, 3)).to(device)
        
        if debug_mode:
            # Set first point velocity for debugging
            self.velocities[0] = torch.tensor([0.0, -0.4, 0.0]).to(device)
            # Initialize random velocities for other points
            velocities_random = torch.randn(num_points-1, 3).to(device)
            # Normalize velocities and scale them
            self.velocities[1:] = velocities_random / torch.norm(velocities_random, dim=1, keepdim=True) * 0.2
        else:
            # Initialize all velocities randomly
            velocities_random = torch.randn(num_points, 3).to(device)
            # Normalize velocities and scale them
            self.velocities = velocities_random / torch.norm(velocities_random, dim=1, keepdim=True) * 0.2

    def _initialize_safe_positions(self):
        """Initialize positions ensuring points are outside the origin radius"""
        positions = torch.zeros((self.num_points, 3)).to(self.device)
        
        if self.debug_mode:
            # Set first point to specified position
            positions[0] = torch.tensor([0.0, 0.8, 0.6]).to(self.device)
            start_idx = 1
        else:
            start_idx = 0
            
        # Initialize remaining points
        for i in range(start_idx, self.num_points):
            while True:
                # Generate random position within bounds
                pos = torch.zeros(3).to(self.device)
                for j in range(3):
                    pos[j] = torch.rand(1).to(self.device) * \
                        (self.bounds[j*2+1] - self.bounds[j*2]) + self.bounds[j*2]
                
                # Check distance from origin
                dist_from_origin = torch.norm(pos)
                if dist_from_origin > self.origin_radius:
                    positions[i] = pos
                    break
                    
        return positions

    def _reflect_velocity_from_origin(self, position, velocity):
        """Compute reflected velocity when hitting origin boundary"""
        # Normalize position vector to get normal vector of sphere surface
        normal = position / torch.norm(position)
        
        # Compute reflection using v_reflected = v - 2(v·n)n
        dot_product = torch.dot(velocity, normal)
        reflection = velocity - 2 * dot_product * normal
        
        return reflection

    def update_positions(self, dt):
        """Update positions based on velocities and handle boundary conditions"""
        # Update positions
        new_positions = self.positions + self.velocities * dt
        
        # Special handling for first point only in debug mode
        if self.debug_mode:
            if new_positions[0, 1] < -0.8:
                new_positions[0, 1] = -0.8
                self.velocities[0, 1] = 0.0
            elif new_positions[0, 1] > 0.8:
                new_positions[0, 1] = 0.8
                self.velocities[0, 1] = -0.5
        
        # Handle origin avoidance for all points (except first in debug mode)
        start_idx = 1 if self.debug_mode else 0
        for i in range(start_idx, self.num_points):
            dist_from_origin = torch.norm(new_positions[i])
            if dist_from_origin < self.origin_radius:
                # Move point to boundary
                new_positions[i] = new_positions[i] * (self.origin_radius / dist_from_origin)
                # Reflect velocity
                self.velocities[i] = self._reflect_velocity_from_origin(new_positions[i], self.velocities[i])
        
        # Handle outer bounds
        for i in range(3):
            # Check lower bound
            mask_low = new_positions[start_idx:, i] < self.bounds[i*2]
            new_positions[start_idx:][mask_low, i] = self.bounds[i*2]
            self.velocities[start_idx:][mask_low, i] *= -1
            
            # Check upper bound
            mask_high = new_positions[start_idx:, i] > self.bounds[i*2+1]
            new_positions[start_idx:][mask_high, i] = self.bounds[i*2+1]
            self.velocities[start_idx:][mask_high, i] *= -1
        
        self.positions = new_positions
        return self.positions

class RobotVelocityController:
    def __init__(self, robot_id, kp=0.8, kd=0.1):
        self.robot_id = robot_id
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.num_joints = 7
        self.prev_error = np.zeros(7)
        
    def compute_velocity_command(self, current_joints, target_joints, dt):
        """Compute joint velocities to reach target configuration using PD control"""
        # Convert to numpy arrays if they're torch tensors
        if torch.is_tensor(current_joints):
            current_joints = current_joints.cpu().numpy()
        if torch.is_tensor(target_joints):
            target_joints = target_joints.cpu().numpy()
            
        # Compute position error
        position_error = target_joints - current_joints
        
        # Compute error derivative (velocity error)
        error_derivative = (position_error - self.prev_error) / dt
        self.prev_error = position_error
        
        # PD control law
        joint_velocities = self.kp * position_error + self.kd * error_derivative

        
        # Clip velocities to reasonable limits
        max_velocity = 2.0  # rad/s
        joint_velocities = np.clip(joint_velocities, -max_velocity, max_velocity)
        
        
        return joint_velocities

class RobotSDFVisualizer:
    def __init__(self, ee_goal, use_gui=True):
        # Initialize environment
        self.env = FrankaEnvironment(gui=use_gui)
        self.physics_client = self.env.client
        self.robot_id = self.env.robot_id
        self.use_gui = use_gui

        # Robot base transform
        self.base_pos = torch.tensor([-0.6, 0.1, 0.6], device='cuda')

        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize robot and SDF model
        self.robot = PandaLayer(self.device, mesh_path="panda_layer/meshes/visual/*.stl")
        self.bp_sdf = BPSDF(
            n_func=8,
            domain_min=-1.0,
            domain_max=1.0,
            robot=self.robot,
            device=self.device
        )
        
        # Load pre-trained model
        CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(CUR_DIR, 'models', 'BP_8.pt')
        self.bp_sdf.model = torch.load(model_path)
        
        # Store goal configuration (in task space)
        self.goal = ee_goal
        
        # Initialize MPPI controller after BP-SDF is properly loaded
        self.mppi_controller = setup_mppi_controller(
            learned_CDF=self.bp_sdf,
            use_GPU=(self.device=='cuda'),
            samples=500,
            initial_horizon=10
        )
        
        # Create goal marker
        # self.create_goal_marker(goal_config)

    def set_robot_configuration(self, joint_angles):
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        
        if len(joint_angles.shape) == 2:
            joint_angles = joint_angles[0]
        
        for i in range(7):
            p.resetJointState(self.robot_id, i, joint_angles[i])

    def get_current_joint_states(self):
        joint_states = []
        for i in range(7):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])
        return torch.tensor(joint_states, device=self.device)

    def create_goal_marker(self, goal_config):
        self.goal_marker = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[0, 1, 0, 0.5]
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
        ee_pos_local = forward_kinematics_batch(joint_angles)
        
        # Transform to world frame
        ee_pos_world = ee_pos_local + self.base_pos
        
        return ee_pos_world

    def run_demo(self, duration=6.0, fps=20):
        """Run demo with MPPI control and record video"""
        print("Starting MPPI demo...")
        time_steps = int(duration * fps)
        dt = 1.0 / fps
        
        # Lists to store distances
        goal_distances = []
        sdf_distances = []
        
        # Setup camera parameters
        width = 1920
        height = 1080
        frames = []
        
        # Initialize state and control sequence
        current_state = self.get_current_joint_states()
        U = torch.zeros((10, 7), device=self.device)
        
        # Get point cloud obstacles
        points_world, points_robot = self.env.get_point_cloud(downsample=True)
        points_world = torch.from_numpy(points_world).float().to(self.device)
        points_robot = torch.from_numpy(points_robot).float().to(self.device)
        
        print("Moving to goal configuration...")
        
        for step in range(time_steps):
            current_state = self.get_current_joint_states()
            
            # Compute current end-effector position and distance to goal
            current_ee_pos = self.get_ee_position(current_state)
            goal_dist = torch.norm(self.goal- current_ee_pos.squeeze())
            goal_distances.append(goal_dist.cpu().numpy())
            
            # Get SDF values
            robot_pose = torch.eye(4).unsqueeze(0).to(self.device)
            sdf, _ = self.bp_sdf.get_whole_body_sdf_batch(
                points_robot,  # Use points in robot frame
                robot_pose,
                current_state.unsqueeze(0),
                self.bp_sdf.model,
                use_derivative=False
            )
            sdf_distances.append(sdf.min().cpu().numpy())

            #print('current state: ', current_state)

            #print(f"Distance to obstacle: {sdf.min().item():.4f}")
            
            # Generate control using MPPI
            sampled_states, states_final, action, U = self.mppi_controller(
                key=None,
                U=U,
                init_state=current_state,
                goal=self.goal,
                obstaclesX=points_robot,  # Use points in robot frame
                safety_margin=-0.2,
                batch_size=200
            )
            
            # Update robot state
            self.set_robot_configuration(current_state + action.squeeze() * dt)
            p.stepSimulation()
            
            # Capture frame with rotating camera
            view_matrix = p.computeViewMatrixFromYawPitchRoll( 
                cameraTargetPosition=[0.0, 0.0, 1.0],  
                distance=2.0,
                yaw=(step / time_steps) * 100,  # Rotating camera
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
            
            # Get image
            _, _, rgb, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to RGB and append to frames
            rgb_array = np.array(rgb)[:, :, :3]
            frames.append(rgb_array)
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{time_steps}")
                print(f"Distance to goal: {goal_dist.item():.4f}")
                print(f"Distance to obstacle: {sdf.min().item():.4f}")
                print("---")
            
            # Optional: early stopping
            if goal_dist < 0.1:
                print("Reached goal!")
                break
                
            # Clear some memory
            torch.cuda.empty_cache()
        
        # Save video using imageio
        print("Saving video...")
        imageio.mimsave('robot_demo.mp4', frames, fps=fps)
        
        print("Demo completed. Plotting distances...")
        
        # Convert lists to numpy arrays
        goal_distances = np.array(goal_distances)
        sdf_distances = np.array(sdf_distances)
        
        # Plot distances
        plot_distances(goal_distances, sdf_distances, obst_radius=0.05, dt=dt)
        
        print("Demo completed!")
        return goal_distances, sdf_distances
        

if __name__ == "__main__":
    # Example usage
    goal_config = torch.tensor([0.2, 0.3, 0.8], device='cuda')
    visualizer = RobotSDFVisualizer(goal_config, use_gui=False)
    goal_distances, sdf_distances = visualizer.run_demo()
