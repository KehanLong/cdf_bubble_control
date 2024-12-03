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
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                 [-0.6, 0.3, 0.6], 
                                 useFixedBase=True)
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
        
        # Store objects for tracking
        self.objects = []
        
        # Add default objects if requested
        if add_default_objects:
            self.add_default_objects()
    
    def add_default_objects(self):
        """Add default obstacles to the environment"""
        # Add cubes
        self.add_obstacle("cube.urdf", [0.0, 0.3, 0.8], scaling=0.35)
        self.add_obstacle("cube.urdf", [0.0, 0.0, 0.75], scaling=0.25)
        self.add_obstacle("cube.urdf", [0.0, -0.2, 0.7], scaling=0.15)
        
        # Add duck
        duck_orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
        self.add_obstacle("duck_vhacd.urdf", [0.4, -0.1, 0.6], 
                         duck_orientation, scaling=3.0)

    def add_obstacle(self, urdf_path, position, orientation=None, scaling=1.0):
        """Add obstacle to the environment"""
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        obj_id = p.loadURDF(urdf_path, position, orientation, globalScaling=scaling)
        self.objects.append(obj_id)
        return obj_id

    def downsample_point_cloud(self, points, voxel_size=0.03, target_points=1000):
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
        """Get filtered point cloud of the environment"""
        # Camera setup
        fov = 60
        aspect = width / height
        near = 0.01
        far = 10
        
        camera_position = [-0.5, -2.0, 3.0]
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
        object_mask = ~np.isin(object_ids, [0, 1])
        mask = depth_mask & object_mask
        
        if not np.any(mask):
            print("No valid depth points found!")
            return np.array([])
        
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
        
        # Add downsampling option
        if downsample:
            points_world = self.downsample_point_cloud(points_world)
        
        return points_world

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

def main():
    """Demo script showing environment usage"""
    env = FrankaEnvironment(gui=True)
    
    # Get point cloud once
    points = env.get_point_cloud(downsample=True)
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

    