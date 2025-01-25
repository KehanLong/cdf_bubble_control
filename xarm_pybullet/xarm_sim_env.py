import pybullet as p
import pybullet_data
import numpy as np
import torch
from robot_sdf import RobotSDF
import time
import os
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
import imageio


@dataclass
class IKSolution:
    joint_angles: np.ndarray
    task_dist: float
    min_sdf: float
    min_cdf: float = 1.0

class XArmEnvironment:
    def __init__(self, gui=True, add_default_objects=True, add_dynamic_obstacles=True):
        """Initialize PyBullet environment with xArm robot"""
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Disable the coordinate axes visualization
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_AXES, 0)
        
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load basic environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_base_pos = [-0.6, 0.0, 0.625]
        self.robot_id = p.loadURDF(os.path.join(self.script_dir, "xarm_description/xarm6_with_gripper.urdf"),
                                 self.robot_base_pos,
                                 useFixedBase=True)
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
        
        # Store objects for tracking
        self.objects = []
        
        # Store dynamic obstacles
        self.dynamic_obstacles = []
        
        # Cache for static point cloud
        self.cached_static_points = None
        
        # Add default objects if requested
        if add_default_objects:
            self.add_default_objects()
            
        # Add dynamic obstacles if requested
        if add_dynamic_obstacles:
            self.add_dynamic_obstacles()
            
        # Initialize SDF model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sdf_model = RobotSDF(device=self.device)
        
        # Set initial joint positions
        self.initial_joint_positions = [0, -0.5, -0.5, 0, 0.95, 0]
        
        # Reset joints to initial positions
        for i in range(6):
            p.resetJointState(
                self.robot_id,
                i+1,
                self.initial_joint_positions[i]
            )

        self.debug_ids = None  # Single ID for all debug points
        


    def add_default_objects(self):
        """Add default obstacles to the environment"""
        # Load bookshelf
        bookshelf_id = p.loadURDF(os.path.join(self.script_dir,
            "obst_urdf/bookshelf.urdf"),
            basePosition=[0.05, 0.25, 0.625],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            globalScaling=0.4
        )
        self.objects.append(bookshelf_id)

    def add_obstacle(self, urdf_path, position, orientation=None, scaling=1.0):
        """Add obstacle to the environment"""
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        obj_id = p.loadURDF(urdf_path, position, orientation, globalScaling=scaling)
        self.objects.append(obj_id)
        return obj_id

    def add_dynamic_obstacles(self, num_obstacles=3):
        """Add dynamic spherical obstacles with different motion patterns"""
        # Define colors for each obstacle
        colors = [
            [1, 0, 0, 0.7],    # Red
            [0, 0, 1, 0.7],    # Blue
            [0.5, 0, 0.5, 0.7] # Purple
        ]
        
        # Vertical moving obstacle
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.04,
            rgbaColor=colors[0]  # Red
        )
        
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0.1, -0.1, 1.1],
        )
        
        self.dynamic_obstacles.append({
            'id': obstacle_id,
            'type': 'vertical',
            'center': [0.0, -0.3, 1.1],  # Middle position
            'amplitude': 0.4,  # +/- 0.3m from center
            'frequency': 0.2,  # Oscillation frequency
            'phase': 0.0,      # Time tracking
        })
        
        # Horizontal moving obstacle
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.04,
            rgbaColor=colors[1]  # Blue
        )
        
        obstacle_id = p.createMultiBody(
            baseMass=0,  # Mass of 0 makes it kinematic
            #baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0.0, 0.1, 1.2],  # Start at middle of horizontal path
        )
        
        self.dynamic_obstacles.append({
            'id': obstacle_id,
            'type': 'horizontal',
            'center': [0.0, 0.1, 1.2],  # Fixed y and z
            'amplitude': 0.3,  # +/- 0.3m in x direction
            'frequency': 0.05,  # Oscillation frequency
            'phase': 0.0,      # Time tracking
        })
        
        # Figure-8 moving obstacle
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.04,
            rgbaColor=colors[2]  # Purple
        )
        
        x = -0.1
        y = -0.3
        z = 0.8
        
        obstacle_id = p.createMultiBody(
            baseMass=0,  # Mass of 0 makes it kinematic
            #baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[x, y, z],
        )
        
        self.dynamic_obstacles.append({
            'id': obstacle_id,
            'type': 'figure8',
            'center': [x, y, z],
            'amplitude': 0.2,
            'frequency': 0.3,
            'phase': 0.0,
        })

    def update_dynamic_obstacles(self, dt):
        """Update positions of dynamic obstacles"""
        for obstacle in self.dynamic_obstacles:
            if obstacle['type'] == 'vertical':
                # Update phase
                obstacle['phase'] += dt * obstacle['frequency']
                
                # Vertical sinusoidal motion
                new_z = obstacle['center'][2] + obstacle['amplitude'] * np.sin(2 * np.pi * obstacle['phase'])
                new_pos = [
                    obstacle['center'][0],
                    obstacle['center'][1],
                    new_z
                ]
                
                # Set vertical velocity
                vz = obstacle['amplitude'] * 2 * np.pi * obstacle['frequency'] * np.cos(2 * np.pi * obstacle['phase'])
                velocity = [0, 0, vz]
                
            elif obstacle['type'] == 'horizontal':
                # Update phase
                obstacle['phase'] += dt * obstacle['frequency']
                
                # Horizontal sinusoidal motion
                new_x = obstacle['center'][0] + obstacle['amplitude'] * np.sin(2 * np.pi * obstacle['phase'])
                new_pos = [
                    new_x,
                    obstacle['center'][1],
                    obstacle['center'][2]
                ]
                
                # Set horizontal velocity
                vx = obstacle['amplitude'] * 2 * np.pi * obstacle['frequency'] * np.cos(2 * np.pi * obstacle['phase'])
                velocity = [vx, 0, 0]
                
            elif obstacle['type'] == 'figure8':
                # Update phase
                obstacle['phase'] += dt * obstacle['frequency']
                
                # Figure-8 pattern (lemniscate of Bernoulli)
                a = obstacle['amplitude']
                t = 2 * np.pi * obstacle['phase']
                
                # Figure-8 coordinates relative to center
                dx = a * np.cos(t) / (1 + np.sin(t)**2)
                dy = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
                
                new_pos = [
                    obstacle['center'][0] + dx,
                    obstacle['center'][1] + dy,
                    obstacle['center'][2]
                ]
                
                # Calculate velocities (derivatives of position)
                vx = -a * (2 * np.sin(t)**3 - np.sin(t)) / (1 + np.sin(t)**2)**2
                vy = a * np.cos(t) * (2 * np.sin(t)**2 - 1) / (1 + np.sin(t)**2)**2
                velocity = [
                    vx * 2 * np.pi * obstacle['frequency'],
                    vy * 2 * np.pi * obstacle['frequency'],
                    0
                ]
            
            # Update position and velocity
            p.resetBasePositionAndOrientation(
                obstacle['id'],
                new_pos,
                p.getQuaternionFromEuler([0, 0, 0])
            )
            
            p.resetBaseVelocity(
                obstacle['id'],
                linearVelocity=velocity,
                angularVelocity=[0, 0, 0]
            )

    def get_static_point_cloud(self, width=320, height=240, downsample=True, min_height=0.6):
        """Get filtered point cloud of static environment (cached)"""
        if self.cached_static_points is not None:
            return self.cached_static_points

        p.removeAllUserDebugItems()
        
        # Camera setup
        fov = 60
        aspect = width / height
        near = 0.01
        far = 10
        
        camera_position = [-1.0, -2.5, 1.8]
        camera_target = [0.0, 0.0, 0.5]
        up_vector = [0.0, 0.0, 1.0]
        
        view_matrix = p.computeViewMatrix(camera_position, camera_target, up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
        # Get camera image
        _, _, _, depth_buffer, seg_mask = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert depth buffer to numpy array
        depth_buffer = np.array(depth_buffer).reshape(height, width)
        seg_mask = np.array(seg_mask).reshape(height, width)
        
        depth_mask = depth_buffer < 0.999
        object_ids = seg_mask & ((1 << 24) - 1)
        object_mask = ~np.isin(object_ids, [self.plane_id, self.robot_id])
        mask = depth_mask & object_mask
        
        if not np.any(mask):
            return torch.zeros((1, 0, 3), device=self.device)
        
        # Convert to 3D points
        z_ndc = 2.0 * depth_buffer - 1.0
        z_eye = 2.0 * near * far / (far + near - z_ndc * (far - near))
        
        rows, cols = np.mgrid[0:height, 0:width]
        x_ndc = (2.0 * cols / width - 1.0) * aspect
        y_ndc = 1.0 - 2.0 * rows / height
        
        x_cam = x_ndc * z_eye * np.tan(np.radians(fov/2))
        y_cam = y_ndc * z_eye * np.tan(np.radians(fov/2))
        z_cam = -z_eye
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)[mask]
        
        if len(points_cam) == 0:
            return torch.zeros((1, 0, 3), device=self.device)
        
        # Transform to world space
        view_matrix = np.array(view_matrix).reshape(4, 4).T
        R = view_matrix[:3, :3].T
        t = view_matrix[:3, 3]
        points_world = (R @ points_cam.T).T - (R @ t)
        
        # Filter points below min_height
        height_mask = points_world[:, 2] >= min_height
        points_world = points_world[height_mask]
        
        distances = np.linalg.norm(points_world - np.array(self.robot_base_pos), axis=1)
        radius_mask = distances > 0.2  # Filter out points within 0.2 radius
        points_world = points_world[radius_mask]
        
        # Add end effector filtering
        end_effector_state = p.getLinkState(self.robot_id, 7)  # Get end effector link state
        ee_distances = np.linalg.norm(points_world - np.array(end_effector_state[0]), axis=1)
        ee_radius_mask = ee_distances > 0.25  # Filter out points within 0.15m of end effector
        points_world = points_world[ee_radius_mask]
        
        # Add downsampling
        if downsample and len(points_world) > 0:
            points_world = self.downsample_point_cloud(points_world)
        
        # Transform to robot base frame
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_mat = np.array(p.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
        points_robot = (points_world - robot_pos) @ robot_mat.T
        
        # Visualize points in PyBullet (red color)
        # for point in points_world:  # Still visualize in world frame
        #     p.addUserDebugPoints([point], [[1, 0, 0]], pointSize=3)
        
        # Cache the processed point cloud
        self.cached_static_points = torch.tensor(points_robot, device=self.device).unsqueeze(0)
        return self.cached_static_points

    def get_dynamic_points(self):
        """Get points representing dynamic obstacles in robot frame with their velocities"""
        if not self.dynamic_obstacles:
            return torch.zeros((0, 3), device=self.device), torch.zeros((0, 3), device=self.device)

        dynamic_points = []
        dynamic_velocities = []
        for obstacle in self.dynamic_obstacles:
            pos, _ = p.getBasePositionAndOrientation(obstacle['id'])
            lin_vel, _ = p.getBaseVelocity(obstacle['id'])
            
            # Generate points on sphere surface using spherical coordinates
            radius = 0.05  # sphere radius
            num_phi = 2     # number of vertical divisions
            num_theta = 2   # number of horizontal divisions
            
            for i in range(num_phi):
                phi = np.pi * (i + 1) / (num_phi + 1)
                for j in range(num_theta):
                    theta = 2 * np.pi * j / num_theta
                    
                    # Convert spherical to Cartesian coordinates
                    x = pos[0] + radius * np.sin(phi) * np.cos(theta)
                    y = pos[1] + radius * np.sin(phi) * np.sin(theta)
                    z = pos[2] + radius * np.cos(phi)
                    
                    dynamic_points.append(np.array([x, y, z]))
                    dynamic_velocities.append(np.array(lin_vel))  # Each point gets obstacle's velocity
        
        dynamic_points = np.array(dynamic_points)
        dynamic_velocities = np.array(dynamic_velocities)
        
        # Transform points and velocities to robot frame
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_mat = np.array(p.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
        dynamic_points_robot = (dynamic_points - robot_pos) @ robot_mat.T
        dynamic_velocities_robot = dynamic_velocities @ robot_mat.T  # Only rotate velocities, no translation
        
        return (torch.tensor(dynamic_points_robot, device=self.device), 
                torch.tensor(dynamic_velocities_robot, device=self.device))

    def get_full_point_cloud(self):
        """Get static and dynamic point clouds with velocities"""
        # Get static points (cached)
        static_points = self.get_static_point_cloud()
        # Static points have zero velocities
        static_velocities = torch.zeros_like(static_points)
        
        # Get dynamic points and their velocities
        dynamic_points, dynamic_velocities = self.get_dynamic_points()
        
        return {
            'static': {
                'points': static_points.squeeze(0),
                'velocities': static_velocities.squeeze(0)
            },
            'dynamic': {
                'points': dynamic_points,
                'velocities': dynamic_velocities
            }
        }

    def downsample_point_cloud(self, points, voxel_size=0.03, target_points=500):
        """
        Downsample point cloud using voxel grid method with adaptive boundaries
        
        Args:
            points: Nx3 array of points
            voxel_size: Size of voxel for initial downsampling
            target_points: Desired number of points after downsampling
        
        Returns:
            Downsampled point cloud as Mx3 array (M ≈ target_points)
        """
        if len(points) == 0:
            return points
        
        voxel_coords = np.floor((points - np.min(points, axis=0)) / voxel_size)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        downsampled_points = points[unique_indices]
        
        if len(downsampled_points) > target_points:
            selected_indices = np.zeros(target_points, dtype=int)
            min_distances = np.full(len(downsampled_points), np.inf)
            
            centroid = np.mean(downsampled_points, axis=0)
            distances = np.linalg.norm(downsampled_points - centroid, axis=1)
            selected_indices[0] = np.argmin(distances)
            
            for i in range(1, target_points):
                last_point = downsampled_points[selected_indices[i-1]]
                distances = np.linalg.norm(downsampled_points - last_point, axis=1)
                
                # Update minimum distances
                min_distances = np.minimum(min_distances, distances)
                
                # Select point with maximum minimum distance
                next_point = np.argmax(min_distances)
                selected_indices[i] = next_point
                
                # Set selected point's distance to 0 to avoid reselection
                min_distances[next_point] = 0
            
            downsampled_points = downsampled_points[selected_indices]
        
        return downsampled_points
    

    def create_goal_marker(self, goal_pos):
        """Create a visible marker for the goal position"""
        self.goal_marker = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0, 1, 0, 0.7]  # Green, semi-transparent
        )
        
        self.goal_visual = p.createMultiBody(
            baseVisualShapeIndex=self.goal_marker,
            basePosition=goal_pos.cpu().numpy(),
            baseOrientation=[0, 0, 0, 1]
        )

    def step(self):
        """Step the simulation"""
        self.update_dynamic_obstacles(1/240.0)
        p.stepSimulation()

    def reset(self):
        """Reset environment to initial state"""
        for obj_id in self.objects:
            p.removeBody(obj_id)
        self.objects = []
        self.add_default_objects()
        
        # Reset dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            p.removeBody(obstacle['id'])
        self.dynamic_obstacles = []
        self.add_dynamic_obstacles()
        
        # Clear cached point cloud
        self.cached_static_points = None

    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect(self.client)

    def set_ik_parameters(self, max_iterations: int = 2000, 
                         threshold: float = 0.03,
                         max_solutions: int = 10):
        """Set IK solver parameters"""
        self.ik_max_iterations = max_iterations
        self.ik_threshold = threshold
        self.ik_max_solutions = max_solutions
    
    def find_ik_solutions(self, 
                         target_pos: np.ndarray,
                         visualize: bool = False,
                         pause_time: float = 1.0,
                         seed: int = None,
                         min_config_distance: float = 0.2  # enable diverse solutions
                         ) -> List[IKSolution]:
        """
        Find multiple IK solutions for a given target position
        
        Args:
            target_pos: Target position in world frame [x, y, z]
            visualize: Whether to visualize solutions
            pause_time: Time to pause between visualizing solutions
            seed: Random seed for reproducibility
            min_config_distance: Minimum distance between configurations in radians
        
        Returns:
            List of valid IK solutions
        """
        if seed is not None:
            np.random.seed(seed)
        
        valid_solutions = []
        gripper_offset = 0.12
        target_world = target_pos + torch.tensor(self.robot_base_pos, device=self.device)
        state_id = p.saveState()
        
        # Expanded set of base orientations
        base_orientations = [
            [0, np.pi, 0],       # Downward
            [0, np.pi/2, 0],     # Horizontal
            [0, 3*np.pi/4, 0],   # 45 degrees
            [0, np.pi/4, 0],     # -45 degrees
            [np.pi/4, np.pi, 0], # Rotated downward
            [-np.pi/4, np.pi, 0],# Another rotation
            [0, 5*np.pi/6, 0],   # More angles
            [0, 2*np.pi/3, 0],
            [np.pi/6, np.pi, 0],
            [-np.pi/6, np.pi, 0],
        ]
        
        # Generate perturbed orientations
        orientations = []
        for base_euler in base_orientations:
            # Add base orientation
            orientations.append(p.getQuaternionFromEuler(base_euler))
            
            # Add perturbed versions
            for _ in range(2):  # Generate 2 perturbations per base orientation
                perturbed = [
                    base_euler[0] + np.random.uniform(-0.2, 0.2),
                    base_euler[1] + np.random.uniform(-0.2, 0.2),
                    base_euler[2] + np.random.uniform(-0.2, 0.2)
                ]
                orientations.append(p.getQuaternionFromEuler(perturbed))
        
        points_robot = self.get_static_point_cloud()
        
        def config_distance(config1, config2):
            diff = np.array(config1) - np.array(config2)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            return np.linalg.norm(diff)
        
        def is_config_diverse(new_config):
            """Check if configuration is sufficiently different from existing solutions"""
            for sol in valid_solutions:
                if config_distance(new_config, sol.joint_angles) < min_config_distance:
                    return False
            return True
        
        # Try each orientation with multiple random seeds
        for orientation in orientations:
            rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            approach_direction = rot_matrix[:, 2]
            adjusted_target = target_world - gripper_offset * torch.tensor(approach_direction, device=self.device)
            
            # Try multiple random seeds for each orientation
            for seed_offset in range(100):  # Try different random seeds per orientation
                # Randomize initial configuration slightly
                random_init = [np.random.uniform(-0.1, 0.1) for _ in range(6)]
                for i in range(6):
                    p.resetJointState(self.robot_id, i+1, random_init[i])
                
                joint_poses = p.calculateInverseKinematics(
                    self.robot_id,
                    endEffectorLinkIndex=7,  # xarm_gripper_base_link
                    targetPosition=adjusted_target,
                    targetOrientation=orientation,
                    maxNumIterations=200,  # Increased from 100
                    residualThreshold=self.ik_threshold
                )
                
                config = joint_poses[:6]
                
                # Skip if not diverse enough
                if not is_config_diverse(config):
                    continue
                
                self._set_robot_configuration(config)
                current_pos_world = self._get_ee_position()
                
                # Add gripper offset to current position for comparison
                current_pos_with_offset = current_pos_world + gripper_offset * approach_direction
                current_pos_base = current_pos_with_offset - np.array(self.robot_base_pos)
                
                task_dist = np.linalg.norm(target_pos.detach().cpu().numpy() - current_pos_base)
                
                if task_dist > self.ik_threshold:
                    continue
                
                min_sdf = 1.0
                if self.sdf_model is not None:
                    config_tensor = torch.tensor(config, device=self.device, dtype=torch.float32)
                    sdf_values = self.sdf_model.query_sdf(
                        points=points_robot,
                        joint_angles=config_tensor.unsqueeze(0),
                        return_gradients=False
                    )
                    min_sdf = sdf_values.min().item()
                
                if min_sdf > 0.05:    # some safety margin for the goal config
                    solution = IKSolution(
                        joint_angles=config,
                        task_dist=task_dist,
                        min_sdf=min_sdf
                    )
                    valid_solutions.append(solution)
                    
                    if visualize:
                        print(f"\nFound solution {len(valid_solutions)}:")
                        print(f"Task distance: {task_dist:.4f}")
                        print(f"Min SDF: {min_sdf:.4f}")
                        print(f"Joint angles: {config}")
                        time.sleep(pause_time)
                    
                    if len(valid_solutions) >= self.ik_max_solutions:
                        break
            
            if len(valid_solutions) >= self.ik_max_solutions:
                break
        
        p.restoreState(state_id)
        p.removeState(state_id)
        
        return valid_solutions
    
    def _set_robot_configuration(self, joint_angles: np.ndarray):
        """Set robot joint angles"""
        for i in range(6):
            p.resetJointState(self.robot_id, i+1, joint_angles[i])
    
    def _get_ee_position(self) -> np.ndarray:
        """Get end-effector position in world frame"""
        link_state = p.getLinkState(self.robot_id, 7)  # xarm_gripper_base_link
        return np.array(link_state[0])

    def print_robot_info(self):
        """Print information about robot joints and links"""
        num_joints = p.getNumJoints(self.robot_id)
        print(f"\nRobot has {num_joints} joints/links:")
        print("-" * 50)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]  # 0=revolute, 1=prismatic, 4=fixed
            
            type_str = "REVOLUTE" if joint_type == p.JOINT_REVOLUTE else \
                      "PRISMATIC" if joint_type == p.JOINT_PRISMATIC else \
                      "FIXED" if joint_type == p.JOINT_FIXED else str(joint_type)
            
            print(f"Link Index: {i}")
            print(f"Joint Name: {joint_name}")
            print(f"Link Name: {link_name}")
            print(f"Joint Type: {type_str}")
            print("-" * 50)

    def capture_snapshot(self, camera_params: Dict, width: int = 1920, height: int = 1080, filename: str = None) -> np.ndarray:
        """Capture a high-quality snapshot from a specific viewpoint"""
        output_dir = "results/snapshots"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f'snapshot_{filename}.png')
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_params.get('target', [0.0, 0.0, 1.5]),
            distance=camera_params.get('distance', 2.5),
            yaw=camera_params.get('yaw', 0),
            pitch=camera_params.get('pitch', -30),
            roll=camera_params.get('roll', 0),
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
        
        img = np.array(rgb)[:, :, :3]
        
        if filename:
            plt.imsave(filename, img)
        
        return img

    def record_trajectory_video(self, trajectory, fps=30, width=1920, height=1080, planner_type=""):
        """Record a video of the trajectory execution"""
        output_dir = "results/videos"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f'trajectory_{planner_type}.mp4')
        
        # Setup camera parameters
        camera_params = {
            'target': [0.0, 0.0, 1.5],
            'distance': 2.5,
            'yaw': 45,  # Angled view
            'pitch': -30,
            'roll': 0
        }
        
        frames = []
        print("\nRecording trajectory video...")
        
        for config in trajectory:
            if isinstance(config, np.ndarray):
                config = torch.tensor(config, device='cuda', dtype=torch.float32)
            
            # Set robot configuration
            self._set_robot_configuration(config)
            p.stepSimulation()
            
            # Capture frame
            frame = self.capture_snapshot(camera_params, width=width, height=height)
            frames.append(frame)
        
        # Save video using imageio
        print(f"Saving video to: {filename}")
        imageio.mimsave(filename, frames, fps=fps)
        print("Video saved successfully!")
        
        return filename

    def capture_trajectory_snapshots(self, trajectory, snapshot_indices=None, planner_type="", views=None):
        """Capture snapshots of the trajectory from multiple viewpoints"""
        if views is None:
            views = [
                {
                    'name': 'front',
                    'params': {'target': [0.0, 0.0, 1.5], 'distance': 2.5, 'yaw': 0, 'pitch': -30}
                },
                {
                    'name': 'side',
                    'params': {'target': [0.0, 0.0, 1.5], 'distance': 2.5, 'yaw': 90, 'pitch': -30}
                },
                {
                    'name': 'top',
                    'params': {'target': [0.0, 0.0, 1.5], 'distance': 2.5, 'yaw': 0, 'pitch': -89}
                }
            ]
        
        if snapshot_indices is None:
            # Take snapshots at start, middle, and end of trajectory
            snapshot_indices = [0, len(trajectory)//2, len(trajectory)-1]
        
        for idx in snapshot_indices:
            config = trajectory[idx]
            if isinstance(config, np.ndarray):
                config = torch.tensor(config, device='cuda', dtype=torch.float32)
            
            # Set robot configuration
            self._set_robot_configuration(config)
            p.stepSimulation()
            
            # Capture from each viewpoint
            for view in views:
                filename = f"snapshot_{planner_type}_{view['name']}_waypoint_{idx}.png"
                self.capture_snapshot(view['params'], filename=filename)
                print(f"Saved snapshot: {filename}")

    def plot_trajectory_metrics(self, goal_distances, sdf_distances, dt, planner_type=""):
        """Plot trajectory metrics and save to file"""
        time_steps = np.arange(len(goal_distances)) * dt
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(time_steps, goal_distances, color='red', linestyle=':', linewidth=3, label='Distance to Goal')
        
        if isinstance(sdf_distances, np.ndarray):
            if sdf_distances.ndim == 1:
                plt.plot(time_steps, sdf_distances, linewidth=3, label='Distance to Obstacle')
            else:
                if sdf_distances.ndim == 3:
                    sdf_distances = sdf_distances.squeeze(1)
                
                num_obstacles = sdf_distances.shape[1]
                for i in range(num_obstacles):
                    if i == 0:
                        plt.plot(time_steps, sdf_distances[:, i], linewidth=3, label='Distance to Obstacles')
                    else:
                        plt.plot(time_steps, sdf_distances[:, i], linewidth=3)
        
        plt.xlabel('Time (s)', fontsize=18)
        plt.ylabel('Distance', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18, loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'distances_plot_{planner_type}.png', dpi=300)
        plt.close()

def main():
    """Demo script showing environment usage with IK"""
    env = XArmEnvironment(gui=True, add_dynamic_obstacles=False)

    # print(env.print_robot_info())
    
    try:
        # Set IK parameters
        env.set_ik_parameters(
            max_iterations=2000,
            threshold=0.05,
            max_solutions=5
        )
        
        # Test different target positions (in robot base frame)
        test_positions = [
            torch.tensor([0.1, -0.5, 0.7], device='cuda'),   # Front right
            torch.tensor([0.7, 0.2, 0.6], device='cuda'),   # Front right
            torch.tensor([0.2, 0.6, 0.7], device='cuda'),   # Front left
            torch.tensor([0.4, 0.0, 1.0], device='cuda'),   # High center
        ]
        
        for target_pos in test_positions:
            print(f"\nTesting target position (robot base frame): {target_pos}")
            
            # Create goal marker (needs world frame)
            env.create_goal_marker(target_pos + torch.tensor(env.robot_base_pos, device='cuda'))
            
            # Find and visualize solutions (pass in robot base frame)
            solutions = env.find_ik_solutions(
                target_pos=target_pos,  # Keep in robot base frame
                visualize=True,
                pause_time=1.0, 
                seed=42
            )
            
            print(f"Found {len(solutions)} valid solutions")
            time.sleep(2)
            
            # Step simulation to show movement
            for _ in range(100):
                env.step()
                time.sleep(0.01)
    
    finally:
        env.close()

if __name__ == "__main__":
    main()
