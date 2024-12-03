import pybullet as p
import pybullet_data
import numpy as np
import time
import os

def create_camera_matrix(fov, aspect, near, far):
    """Create camera projection matrix"""
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2)
    projection_matrix = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ])
    return projection_matrix

def camera_to_world(point_camera, view_matrix):
    """
    Transform point from camera frame to world frame
    point_camera: [x, y, z] in camera frame
    view_matrix: 4x4 view matrix (world to camera)
    """
    # Extract rotation (upper 3x3) and translation from view matrix
    R = view_matrix[:3, :3]  # 3x3 rotation
    t = view_matrix[:3, 3]   # translation
    
    # To go from camera to world:
    # 1. Inverse rotation: R.T (since R is orthogonal)
    # 2. Inverse translation: -R.T @ t
    point_world = R.T @ point_camera - R.T @ t
    
    return point_world

def world_to_camera(point_world, view_matrix):
    """Transform point from world to camera frame"""
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    point_camera = R @ point_world + t
    return point_camera

def get_point_cloud(width=320, height=240):
    # Camera setup
    fov = 60
    aspect = width / height
    near = 0.01
    far = 10
    
    camera_position = [0.0, -2.0, 3.0]
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
    
    # Debug printing - these lines are just for visualization/debugging
    # unique_ids = np.unique(seg_mask & ((1 << 24) - 1))  
    # unique_links = np.unique(seg_mask >> 24)  
    
    # print("\nSegmentation Analysis:")
    # print("Unique object IDs:", unique_ids)  # Shows: [0(ground), 1(robot), 2(table), 3(cube), 4(duck), 16777215(background)]
    # print("Unique link indices:", unique_links)  # Shows: [-1(background), 0(base), 1-11(robot links)]
    
    # Actual point cloud filtering code
    depth_mask = depth_buffer < 0.9999  # True for valid depth points (not at infinity)
    
    # Extract object IDs from segmentation mask
    object_ids = seg_mask & ((1 << 24) - 1)  # Get lower 24 bits containing object ID
    
    # Extract link indices from segmentation mask
    # Using 100 instead of 24 still works because it makes all non-robot points zero
    link_indices = seg_mask >> 24  
    
    # Create mask that keeps only objects we want to see
    # False for ground (ID 0) and robot base (ID 1), True for everything else
    object_mask = ~np.isin(object_ids, [0, 1])
    
    # Create mask that removes robot links
    # True only for points with link_index = 0 (non-robot points)
    link_mask = link_indices == 0
    
    # Combine all masks
    mask = depth_mask & object_mask 
    
    # print("\nDepth buffer analysis:")
    # print("Min:", np.min(depth_buffer))
    # print("Max:", np.max(depth_buffer))
    # print("Mean:", np.mean(depth_buffer))
    # print("Valid points:", np.sum(mask))
    
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
    
    
    return points_world

def main():
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane and table
    plane_id = p.loadURDF("plane.urdf")
    table_pos = [0, 0, 0]
    robot_id = p.loadURDF("franka_panda/panda.urdf", [-0.6, 0.3, 0.6], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", basePosition=table_pos)
    
    # Add some objects on the table
    objects = []
    # Add a cube
    cube_id = p.loadURDF("cube.urdf", [0.0, 0.3, 0.8], globalScaling=0.35)
    objects.append(cube_id)

    cube_id_2 = p.loadURDF("cube.urdf", [0.0, 0., 0.75], globalScaling=0.25)
    objects.append(cube_id_2)

    cube_id_3 = p.loadURDF("cube.urdf", [0.0, -0.2, 0.7], globalScaling=0.15)
    objects.append(cube_id_3)
    
    # Add a sphere (using duck.urdf as it's a small object)
    duck_orientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])  
    sphere_id = p.loadURDF("duck_vhacd.urdf", 
                          [0.4, -0.1, 0.6],  # position
                          duck_orientation,  # orientation
                          globalScaling=3.0)
    objects.append(sphere_id)
    
    # Print object IDs for debugging
    print("\nObject IDs:")
    print(f"Plane ID: {plane_id}")
    print(f"Robot ID: {robot_id}")
    print(f"Table ID: {table_id}")
    print(f"Cube ID: {cube_id}")
    print(f"Sphere ID: {sphere_id}")
    
    while True:
        # Step simulation to let objects settle
        p.stepSimulation()
        
        points = get_point_cloud()
        p.removeAllUserDebugItems()
        
        if len(points) > 0:
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]
            
            for point in points:
                p.addUserDebugPoints([point], [[1,0,0]], pointSize=3)
        
        time.sleep(1)

if __name__ == "__main__":
    main()

    