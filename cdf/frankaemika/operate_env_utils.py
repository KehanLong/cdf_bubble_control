import pybullet as p
import pybullet_data
import numpy as np
import time

class FrankaEnvironment:
    def __init__(self, gui=True, add_default_objects=True):
        """Initialize PyBullet environment with Franka robot"""
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load basic environment
        self.plane_id = p.loadURDF("plane.urdf")

        self.robot_base_pos = [-0.6, 0.0, 0.625]

        # Load Franka robot (fixed base)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                 self.robot_base_pos,  
                                 useFixedBase=True)
        
        # Disable the gripper joints and make gripper components invisible
        # Joint indices: 7,8 for finger joints, and the hand link
        for joint_id in [7, 8]:  # Gripper finger joint indices
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=0
            )

        # Make gripper components invisible
        # This includes the hand and both fingers
        for link_name in ["panda_hand", "panda_leftfinger", "panda_rightfinger"]:
            link_id = -1
            for i in range(p.getNumJoints(self.robot_id)):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[12].decode('utf-8') == link_name:
                    link_id = i
                    p.changeVisualShape(
                        self.robot_id,
                        link_id,
                        rgbaColor=[0, 0, 0, 0]  # Fully transparent
                    )
        
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
        
        # Store objects for tracking
        self.objects = []
        self.duck_id = None  # Initialize duck_id
        self.bookshelf_id = None  # Initialize bookshelf_id
        
        # Add default objects if requested
        if add_default_objects:
            self.add_default_objects()
    
    def add_default_objects(self):
        """Add default obstacles to the environment"""
        # Load bookshelf
        self.bookshelf_id = p.loadURDF(
            "obst_urdf/bookshelf.urdf",
            basePosition=[0.2, 0.2, 0.625],  # Position on table
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
            globalScaling=0.4  # Scale down the bookshelf
        )
        self.objects.append(self.bookshelf_id)
        
        # Add duck and store its ID
        duck_orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
        self.duck_id = self.add_obstacle("duck_vhacd.urdf", [0.4, 0.15, 0.625], 
                                       duck_orientation, scaling=2.0)

    def add_obstacle(self, urdf_path, position, orientation=None, scaling=1.0):
        """Add obstacle to the environment"""
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        obj_id = p.loadURDF(urdf_path, position, orientation, globalScaling=scaling)
        self.objects.append(obj_id)
        return obj_id

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
        
        # Compute point cloud bounds
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        
        # Print bounds for debugging
        
        # Compute voxel coordinates with offset to ensure positive values
        voxel_coords = np.floor((points - min_bounds) / voxel_size)
        
        # Get unique voxels and their first point
        _, unique_indices, counts = np.unique(voxel_coords, axis=0, 
                                            return_index=True, 
                                            return_counts=True)
        
        
        downsampled_points = points[unique_indices]
        
        # If still too many points, use FPS (Farthest Point Sampling)
        if len(downsampled_points) > target_points:
            # Initialize arrays for FPS
            selected_indices = np.zeros(target_points, dtype=int)
            min_distances = np.full(len(downsampled_points), np.inf)
            
            # Start with the centroid point
            centroid = np.mean(downsampled_points, axis=0)
            distances = np.linalg.norm(downsampled_points - centroid, axis=1)
            first_point = np.argmin(distances)
            selected_indices[0] = first_point
            
            # Iteratively select points
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

    def get_point_cloud(self, width=320, height=240, downsample=True, min_height=0.6):
        """
        Get filtered point cloud of the environment
        
        Returns:
            tuple: (points_world, points_robot) - Points in world frame and robot frame
        """
        # Camera setup
        fov = 60
        aspect = width / height
        near = 0.01
        far = 10
        
        camera_position = [-1., -2.5, 1.8]
        camera_target = [0.0, 1.0, 0.0]
        up_vector = [0.0, 0.0, 1.0]
        
        view_matrix = np.array(p.computeViewMatrix(camera_position, camera_target, up_vector)).reshape(4,4).T
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
        # Get camera image with segmentation mask
        _, _, _, depth_buffer, seg_mask = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix.T.ravel(),
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            shadow=0
        )
        
        depth_buffer = np.array(depth_buffer).reshape(height, width)
        seg_mask = np.array(seg_mask).reshape(height, width)
        
        # Filtering masks
        depth_mask = depth_buffer < 0.9999
        object_ids = seg_mask & ((1 << 24) - 1)
        link_indices = seg_mask >> 24
        
        # Update object mask to exclude plane, robot, and duck
        object_mask = ~np.isin(object_ids, [self.plane_id, self.robot_id, self.duck_id])
        mask = depth_mask & object_mask
        
        if not np.any(mask):
            print("No valid depth points found!")
            return np.array([]), np.array([])
        
        # Convert normalized depth to metric depth
        z_ndc = 2.0 * depth_buffer - 1.0
        z_eye = 2.0 * near * far / (far + near - z_ndc * (far - near))
        
        # Generate pixel coordinates
        rows, cols = np.mgrid[0:height, 0:width]
        x_ndc = (2.0 * cols / width - 1.0) * aspect
        y_ndc = 1.0 - 2.0 * rows / height
        
        # Convert to camera space
        x_cam = x_ndc * z_eye * np.tan(np.radians(fov/2))
        y_cam = y_ndc * z_eye * np.tan(np.radians(fov/2))
        z_cam = -z_eye
        
        # Stack and reshape
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        points_cam = points_cam.reshape(-1, 3)
        
        # Apply mask
        points_cam = points_cam[mask.reshape(-1)]
        
        # Transform to world space
        R = view_matrix[:3, :3].T
        t = view_matrix[:3, 3]
        points_world = (R @ points_cam.T).T - (R @ t)
        
        # Filter points below min_height
        height_mask = points_world[:, 2] >= min_height
        points_world = points_world[height_mask]
        
        # Filter points near robot base
        distances = np.linalg.norm(points_world - np.array(self.robot_base_pos), axis=1)
        radius_mask = distances > 0.3  # Filter out points within 0.3 radius
        points_world = points_world[radius_mask]
        
        # Add downsampling option
        if downsample:
            points_world = self.downsample_point_cloud(points_world)
        
        # Transform to robot frame
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_mat = np.array(p.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
        points_robot = (points_world - robot_pos) @ robot_mat.T
        
        return points_world, points_robot

    def reset(self):
        """Reset environment to initial state"""
        for obj_id in self.objects:
            p.removeBody(obj_id)
        self.objects = []

    def step(self):
        """Step the simulation"""
        p.stepSimulation()

    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect(self.client)

    def solve_ik(self, target_pos, target_orn=None, robot_base_pos=None):
        """
        Solve inverse kinematics for the Franka robot.
        
        Args:
            target_pos (list or np.ndarray): Target position [x, y, z] in world frame
            target_orn (list or np.ndarray, optional): Target orientation as quaternion [x, y, z, w]. 
                                                      If None, only position is considered.
            robot_base_pos (list or np.ndarray, optional): Robot base position. 
                                                          If None, uses default base position.
        
        Returns:
            np.ndarray: Joint angles that achieve the target pose, or None if no solution found
        """
        # Add a visual marker for the target position (at the beginning of the method)
        #p.addUserDebugLine([target_pos[0], target_pos[1], 0], target_pos, [1, 0, 0], lineWidth=3)  # Vertical red line
        p.addUserDebugPoints([target_pos], [[0, 1, 0]], pointSize=10)  # Green point at target
        
        if robot_base_pos is None:
            robot_base_pos = self.robot_base_pos
        
        if target_orn is None:
            # Default orientation: gripper pointing downward
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Get end effector link index (link 7 for Franka)
        end_effector_index = 6
        
        # Store current joint states
        current_joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        # Define joint limits for Franka
        joint_limits = [
            [-2.8973, 2.8973],  # joint 1
            [-1.7628, 1.7628],  # joint 2
            [-2.8973, 2.8973],  # joint 3
            [-3.0718, -0.0698], # joint 4
            [-2.8973, 2.8973],  # joint 5
            [-0.0175, 3.7525],  # joint 6
            [-2.8973, 2.8973]   # joint 7
        ]
        
        # Try multiple initial configurations
        best_solution = None
        min_error = float('inf')
        n_attempts = 5
        
        for attempt in range(n_attempts):
            # Generate random initial configuration within joint limits
            if attempt == 0:
                # First attempt: use current configuration
                initial_guess = current_joint_states
            else:
                # Subsequent attempts: random configuration within limits
                initial_guess = [np.random.uniform(limit[0], limit[1]) for limit in joint_limits]
                
            # Set initial configuration
            for i in range(7):
                p.resetJointState(self.robot_id, i, initial_guess[i])
            
            # Calculate IK
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                end_effector_index,
                target_pos,
                target_orn,
                lowerLimits=[limit[0] for limit in joint_limits],
                upperLimits=[limit[1] for limit in joint_limits],
                jointRanges=[limit[1] - limit[0] for limit in joint_limits],
                restPoses=initial_guess,
                maxNumIterations=100,
                residualThreshold=1e-4
            )
            
            # Verify IK solution
            for i in range(7):
                p.resetJointState(self.robot_id, i, joint_poses[i])
            
            # Get resulting end effector position
            link_state = p.getLinkState(self.robot_id, end_effector_index)
            achieved_pos = link_state[0]
            achieved_orn = link_state[1]
            
            # Calculate position error
            pos_error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
            
            # Update best solution if this is better
            if pos_error < min_error:
                min_error = pos_error
                best_solution = joint_poses
                
            # If error is acceptable, break early
            if pos_error < 0.05:  # 5cm threshold
                break
        
        # Reset robot to original joint states
        for i in range(7):
            p.resetJointState(self.robot_id, i, current_joint_states[i])
        
        # Check if best solution is acceptable
        if min_error > 0.05:  # 5cm threshold
            print(f"Warning: Best IK solution found but position error is {min_error:.3f} meters")
            return np.array(best_solution)
        
        return np.array(best_solution)

    def test_ik(self, goal_pos):
        """Test the IK solver with a specific target"""
        joint_angles = self.solve_ik(goal_pos)
        joint_angles = np.array([-0.1219711, 1.08143656, 1.17792112, -0.19790296, 1.79101167, 2.16419901, 0.])
        if joint_angles is not None:
            print(f"Found solution: {joint_angles}")
            
            # Store original joint positions
            original_joints = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
            
            # Move to IK solution
            for i in range(7):
                p.resetJointState(self.robot_id, i, joint_angles[i])
            
            # Add visual marker at target position
            p.addUserDebugPoints([goal_pos], [[1,0,0]], pointSize=5)
            
            input("Press Enter to reset robot position...")
            
            # Reset to original position
            for i in range(7):
                p.resetJointState(self.robot_id, i, original_joints[i])
        else:
            print("No IK solution found")

def main():
    """Demo script showing environment usage"""
    env = FrankaEnvironment(gui=True)
    
    # Test IK with a specific target
    target_pos = [0.1, 0.0, 1.25]
    env.test_ik(target_pos)
    
    # Get point cloud once
    points, _ = env.get_point_cloud(downsample=True)
    print(f"Captured point cloud shape: {points.shape}")
    
    try:
        while True:
            env.step()
            
            # Visualize the same point cloud repeatedly
            p.removeAllUserDebugItems()
            
            if len(points) > 0:
                for point in points:
                    p.addUserDebugPoints([point], [[1,0,0]], pointSize=3)
            
            time.sleep(1)
    
    finally:
        env.close()

if __name__ == "__main__":
    main()
