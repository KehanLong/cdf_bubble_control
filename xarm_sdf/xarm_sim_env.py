import pybullet as p
import pybullet_data
import numpy as np
import torch
from robot_sdf import RobotSDF
import time


class XArmEnvironment:
    def __init__(self, gui=True, add_default_objects=True):
        """Initialize PyBullet environment with xArm robot"""
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load basic environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_base_pos = [-0.6, 0.0, 0.625]
        self.robot_id = p.loadURDF("xarm_description/xarm6_with_gripper.urdf",  # or xarm6_robot.urdf
                                 self.robot_base_pos,
                                 useFixedBase=True)
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
        
        # Store objects for tracking
        self.objects = []
        
        # Add default objects if requested
        if add_default_objects:
            self.add_default_objects()
            
        # Initialize SDF model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sdf_model = RobotSDF(device=self.device)
    
    def add_default_objects(self):
        """Add default obstacles to the environment"""
        # Load bookshelf
        bookshelf_id = p.loadURDF(
            "obst_urdf/bookshelf.urdf",
            basePosition=[0.05, 0.2, 0.625],
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

    def get_point_cloud(self, width=320, height=240, downsample=True, min_height=0.6):
        """Get filtered point cloud of the environment"""
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
        
        return torch.tensor(points_robot, device=self.device).unsqueeze(0)

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

    def reset(self):
        """Reset environment to initial state"""
        for obj_id in self.objects:
            p.removeBody(obj_id)
        self.objects = []
        self.add_default_objects()

    def step(self):
        """Step the simulation"""
        p.stepSimulation()

    def close(self):
        """Disconnect from PyBullet"""
        p.disconnect(self.client)

def main():
    """Demo script showing environment usage"""
    env = XArmEnvironment(gui=True)
    try:
        while True:
            env.step()
            points = env.get_point_cloud()
            print(points.shape)
            time.sleep(0.01)
    finally:
        env.close()

if __name__ == "__main__":
    main()