import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import imageio
from load_and_query_points import SDFQueryHelper
from cbf_qp_controller import CbfQpController
from cbf_dro_controller import CbfDroController



def plot_distances(goal_distances, estimated_obstacle_distances, obst_radius, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot distance to goal as blue dotted line
    plt.plot(time_steps, goal_distances, color='red', linestyle=':', linewidth=3, label='Distance to Goal')

    # Remove the extra dimension if it exists
    if estimated_obstacle_distances.ndim == 3:
        estimated_obstacle_distances = estimated_obstacle_distances.squeeze(1)
    
    num_obstacles = estimated_obstacle_distances.shape[1]


    for i in range(num_obstacles):
        if i == 0:
            plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3, label='Distance to Obstacles')
        else:
            plt.plot(time_steps, estimated_obstacle_distances[:, i], linewidth=3)
    
    plt.axhline(y=obst_radius, color='black', linestyle='--', linewidth=3, label='Safety Margin')
    
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Distance', fontsize=18)
    # plt.title('Distances to Goal and Estimated Distances to Obstacles over Time')

    # Set tick label size to 16
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18, loc='upper right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

class PointTrajectoryManager:
    def __init__(self, num_points=10, device='cuda', bounds=None):
        self.num_points = num_points
        self.device = device
        
        # Define bounds for position [min_x, max_x, min_y, max_y, min_z, max_z]
        self.bounds = bounds if bounds is not None else torch.tensor([-1.5, 1.5, -1.5, 1.5, 0, 1.5])
        
        # Initialize random positions within bounds
        self.positions = torch.zeros((num_points, 3)).to(device)
        for i in range(3):
            self.positions[:, i] = torch.rand(num_points).to(device) * \
                (self.bounds[i*2+1] - self.bounds[i*2]) + self.bounds[i*2]
        
        # Initialize random velocities with unit norm
        velocities = torch.randn(num_points, 3).to(device)
        # Normalize velocities and scale them (0.5 means max speed of 0.5 units per second)
        self.velocities = velocities / torch.norm(velocities, dim=1, keepdim=True) * 0.2

        # zero velocities (static obstacles for testing)
        # self.velocities = torch.zeros_like(self.velocities)
        
    def update_positions(self, dt):
        """Update positions based on velocities and handle boundary conditions"""
        # Update positions
        new_positions = self.positions + self.velocities * dt
        
        # Check bounds and reflect velocities if needed
        for i in range(3):
            # Check lower bound
            mask_low = new_positions[:, i] < self.bounds[i*2]
            new_positions[mask_low, i] = self.bounds[i*2]
            self.velocities[mask_low, i] *= -1
            
            # Check upper bound
            mask_high = new_positions[:, i] > self.bounds[i*2+1]
            new_positions[mask_high, i] = self.bounds[i*2+1]
            self.velocities[mask_high, i] *= -1
        
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
    def __init__(self, goal_config, use_gui=True, controller_mode='cbf_qp'):
        # Initialize PyBullet
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.goal_config = goal_config
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load Panda robot and fix its base
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Initialize SDF query helper
        self.sdf_helper = SDFQueryHelper(device='cuda')
        
        # Initialize point trajectory manager
        self.point_manager = PointTrajectoryManager(
            num_points=10,
            device='cuda',
            bounds=torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.2, 1.2])  # Adjusted bounds for robot workspace
        )
        
        # Initialize both controllers
        self.pd_controller = RobotVelocityController(self.robot_id, kp=2.0, kd=0.1)
        self.controller_mode = controller_mode
        if controller_mode == 'cbf_qp':
            self.safety_controller = CbfQpController(p1=1.0, cbf_rate=0.8)
        elif controller_mode == 'dro_cbf_qp':
            self.safety_controller = CbfDroController(p1=1.0, cbf_rate=0.8, wasserstein_r=0.005, epsilon=0.1)
        else:
            raise ValueError(f"Unknown controller mode: {controller_mode}")
        
        # Create goal marker
        self.create_goal_marker(goal_config)

    def set_robot_configuration(self, joint_angles):
        """Set robot joint angles in PyBullet"""
        # Handle both 1D and 2D arrays/tensors
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        
        # If 2D array, take first row
        if len(joint_angles.shape) == 2:
            joint_angles = joint_angles[0]
        
        # Now joint_angles is 1D array
        for i in range(7):
            p.resetJointState(self.robot_id, i, joint_angles[i])

    def generate_random_joint_angles(self):
        """Generate random joint angles within safe limits"""
        # Define joint limits (approximate for Panda robot)
        joint_limits = torch.tensor([
            [-2.8973, 2.8973],  # joint 1
            [-1.7628, 1.7628],  # joint 2
            [-2.8973, 2.8973],  # joint 3
            [-3.0718, -0.0698], # joint 4
            [-2.8973, 2.8973],  # joint 5
            [-0.0175, 3.7525],  # joint 6
            [-2.8973, 2.8973]   # joint 7
        ]).to(self.sdf_helper.device)
        
        # Generate random values between 0 and 1
        random_values = torch.rand(1, 7).to(self.sdf_helper.device)
        
        # Scale random values to joint limits
        joint_range = joint_limits[:, 1] - joint_limits[:, 0]
        joint_angles = joint_limits[:, 0] + random_values * joint_range
        
        return joint_angles

    def create_visual_sphere(self, radius=0.02, rgba=[1,0,0,1]):
        """Create a visual sphere shape for points"""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba
        )
        return visual_shape_id

    def initialize_point_markers(self, initial_points):
        """Initialize sphere markers for points at their correct initial positions
        Args:
            initial_points: torch tensor of shape (num_points, 3) with initial positions
        """
        self.sphere_ids = []
        points_np = initial_points.cpu().numpy()
        
        for point in points_np:
            visual_shape_id = self.create_visual_sphere(radius=0.05)
            sphere_id = p.createMultiBody(
                baseMass=0,  # Static object
                baseVisualShapeIndex=visual_shape_id,
                basePosition=point,  # Use actual initial position
            )
            self.sphere_ids.append(sphere_id)

    def update_point_positions(self, points, sdf_values):
        """Update positions of sphere markers"""
        points_np = points.cpu().numpy()
        
        # Find the index of the point with smallest SDF value
        min_sdf_idx = torch.argmin(sdf_values).item()
        
        for i, (sphere_id, point) in enumerate(zip(self.sphere_ids, points_np)):
            # Color red only for the point with smallest SDF value, blue for others
            color = [1, 0, 0, 1] if i == min_sdf_idx else [0, 0, 1, 1]
            p.changeVisualShape(sphere_id, -1, rgbaColor=color)
            
            # Update position (orientation doesn't matter for spheres)
            p.resetBasePositionAndOrientation(sphere_id, point, [0, 0, 0, 1])

    def get_current_joint_states(self):
        """Get current joint positions"""
        joint_states = []
        for i in range(7):
            joint_states.append(p.getJointState(self.robot_id, i)[0])
        return np.array(joint_states)

    def create_goal_marker(self, goal_config):
        """Create a visual marker (star) at the goal end-effector position"""
        # First get the end-effector position for goal configuration
        # Store current state
        current_states = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        # Temporarily set robot to goal configuration
        for i in range(7):
            p.resetJointState(self.robot_id, i, goal_config[i])
        
        # Get end effector link state (link 7 for Panda)
        ee_state = p.getLinkState(self.robot_id, 6)  # 6 is the end effector link index
        ee_pos = ee_state[0]  # Position of end effector
        
        # Reset robot to original state
        for i in range(7):
            p.resetJointState(self.robot_id, i, current_states[i])
        
        # Create visual shape for goal marker (using a sphere for now, could be modified to star)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.02, 0.02, 0.02],  # Size in each dimension
            rgbaColor=[1, 0, 1, 1]
        )
        
        # Create multibody for the marker
        self.goal_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=ee_pos,
        )

    def run_demo(self, duration=10.0, fps=50):
        """Run a demo with velocity control"""
        print("Starting demo...")
        time_steps = int(duration * fps)
        dt = 1.0 / fps
        
        # Lists to store distances
        goal_distances = []
        sdf_distances = []
        
        # Get initial points and initialize markers
        points = self.point_manager.positions
        self.initialize_point_markers(points)  # Pass initial points to marker initialization
        
        # Setup camera parameters
        width = 1920
        height = 1080
        # List to store frames
        frames = []

        print("Moving to goal configuration...")
        
        for step in range(time_steps):
            # Get current joint states
            current_joints = self.get_current_joint_states()
            
            # Store distance to goal
            goal_dist = np.linalg.norm(current_joints - self.goal_config)
            goal_distances.append(goal_dist)
            
            # Get nominal control from PD controller
            nominal_velocities = self.pd_controller.compute_velocity_command(current_joints, self.goal_config, dt)
            
            # Get current point velocities
            point_velocities = self.point_manager.velocities  # Shape: (num_points, 3)
            
            # Get SDF values and gradients for current points
            robot_pose = torch.eye(4).unsqueeze(0).to(self.sdf_helper.device)
            sdf, sdf_grad = self.sdf_helper.query_sdf(points, robot_pose, 
                torch.from_numpy(current_joints).unsqueeze(0).to(self.sdf_helper.device))
            
            # Central difference for gradients w.r.t obstacles motions
            epsilon = 0.001
            sdf_spatial_grad = torch.zeros_like(points)
            for i in range(3):  # x, y, z
                points_plus = points.clone()
                points_minus = points.clone()
                points_plus[:, i] += epsilon
                points_minus[:, i] -= epsilon
                
                sdf_plus, _ = self.sdf_helper.query_sdf(points_plus, robot_pose,
                    torch.from_numpy(current_joints).unsqueeze(0).to(self.sdf_helper.device))
                sdf_minus, _ = self.sdf_helper.query_sdf(points_minus, robot_pose,
                    torch.from_numpy(current_joints).unsqueeze(0).to(self.sdf_helper.device))
                    
                sdf_spatial_grad[:, i] = (sdf_plus - sdf_minus) / (2 * epsilon)
            
            # Compute cbf_t_grad for all points
            point_velocities_np = point_velocities.cpu().numpy()
            spatial_grad_np = sdf_spatial_grad.cpu().numpy()
            cbf_t_grads = np.sum(spatial_grad_np * point_velocities_np, axis=1)  # Shape: (num_points,)
            
            # Find minimum points considering both terms
            alpha_h = self.safety_controller.rateh  # CBF parameter
            combined_values = sdf.cpu().numpy() * alpha_h + cbf_t_grads
            
            # Get indices of 5 smallest values
            min_indices = np.argpartition(combined_values, 5)[:5]
            
            if self.controller_mode == 'cbf_qp':
                # Original CBF-QP control - use single minimum point
                min_idx = min_indices[0]  # Take the absolute minimum
                sdf_val = sdf[min_idx].cpu().numpy()
                sdf_grad_min = sdf_grad[min_idx].cpu().numpy()
                cbf_t_grad_min = cbf_t_grads[min_idx]
                
                safe_velocities = self.safety_controller.generate_control(
                    nominal_velocities,
                    sdf_val - 0.05,  # h = sdf - safety_margin
                    sdf_grad_min,
                    cbf_t_grad_min
                )
            else:  # dro_cbf_qp
                # Use 5 most critical points for DRO
                cbf_h_samples = (sdf[min_indices].cpu().numpy() - 0.05)  # Apply safety margin
                cbf_h_grad_samples = sdf_grad[min_indices].cpu().numpy()
                cbf_t_grad_samples = cbf_t_grads[min_indices]
                
                safe_velocities = self.safety_controller.generate_control(
                    nominal_velocities,
                    cbf_h_samples,
                    cbf_h_grad_samples,
                    cbf_t_grad_samples
                )

            print('velocities_difference', safe_velocities - nominal_velocities)
            print('distance_to_points:', sdf)

            # Apply safe velocities to robot
            for i in range(7):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=safe_velocities[i],
                    force=100
                )
            
            # Update moving points (obstacles)
            points = self.point_manager.update_positions(dt)
            robot_pose = torch.eye(4).unsqueeze(0).to(self.sdf_helper.device)
            sdf, _ = self.sdf_helper.query_sdf(points, robot_pose, torch.from_numpy(current_joints).unsqueeze(0).to(self.sdf_helper.device))
            
            # Store SDF values
            sdf_distances.append(sdf.cpu().numpy())
            
            self.update_point_positions(points, sdf)

            # Step simulation
            p.stepSimulation()
            time.sleep(1/fps)

            # Capture frame
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=2.0,
                yaw= (step / time_steps) * 100,   # slower rotation of the camera
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


        # Save video using imageio
        imageio.mimsave('robot_demo.mp4', frames, fps=fps)

        print("Demo completed. Plotting distances...")
        
        # Convert lists to numpy arrays
        goal_distances = np.array(goal_distances)
        sdf_distances = np.array(sdf_distances)
        
        # Plot distances
        plot_distances(goal_distances, sdf_distances, obst_radius=0.05, dt=dt)
        
        # Clean up
        for sphere_id in self.sphere_ids:
            p.removeBody(sphere_id)
        p.removeBody(self.goal_marker_id)  # Remove goal marker
        p.disconnect()
        
        
        return goal_distances, sdf_distances

if __name__ == "__main__":


    goal_config = np.array([1.68431763,  0.29743382, -0.65842076 ,-1.87699534, -2.26396217,  1.34391705,
                                   0.20779162], dtype=np.float32)
    
    # Create visualizer with specified controller mode
    visualizer = RobotSDFVisualizer(goal_config, use_gui=False, controller_mode='dro_cbf_qp')
    goal_dists, sdf_dists = visualizer.run_demo(duration=10.0)