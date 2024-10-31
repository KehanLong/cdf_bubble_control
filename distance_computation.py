import os
import pybullet as p
import pybullet_data
import time


def compute_point_to_robot_distance(robot_id, point, joint_angles=None, visualize=False):
    """
    Compute the minimum distance from a point to any part of the robot.
    
    Args:
        robot_id: PyBullet body ID for the robot
        point: [x, y, z] coordinates of the query point
        joint_angles: Optional list of joint angles. If provided, robot will be set to this configuration
        visualize: If True, draw a line showing the closest points
    
    Returns:
        distance: Minimum distance from point to robot
    """
    if joint_angles is not None:
        for i, angle in enumerate(joint_angles):
            p.resetJointState(robot_id, i, angle)
    
    # Create a temporary sphere at the query point
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
    point_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        basePosition=point
    )
    
    closest_points = p.getClosestPoints(robot_id, point_body, distance=100.0)
    min_distance = float('inf')
    closest_point_on_robot = None
    closest_point_on_sphere = None
    
    if closest_points:
        min_distance = closest_points[0][8]
        closest_point_on_robot = closest_points[0][5]
        closest_point_on_sphere = closest_points[0][6]
        
        if visualize:
            # Remove any existing debug lines (if any)
            p.removeAllUserDebugItems()
            # Draw a line between the closest points (green color)
            p.addUserDebugLine(
                closest_point_on_robot,
                closest_point_on_sphere,
                lineColorRGB=[0, 1, 0],
                lineWidth=2,
                lifeTime=0  # 0 means permanent until removed
            )
    
    p.removeBody(point_body)
    p.removeCollisionShape(collision_shape)
    
    return min_distance

def test_distance_computation():
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    
    # Load the robot URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "xarm", "xarm6_robot.urdf")
    robotId = p.loadURDF(urdf_path, useFixedBase=1)
    
    # Test some points with known distances
    test_points = [
        [0.5, 0.0, 0.5],   # Point somewhat in front of robot
        [0.3, 0.3, 0.3],   # Point closer to robot
        [1.0, 1.0, 1.0],   # Point far from robot
        [0.0, 0.0, 0.0],   # Origin point (should be very close to robot base)
        [0.2, 0.0, 0.0],   # Point near robot base
    ]
    
    # Set robot to home configuration
    home_angles = [0.0] * 6  # Assuming 6-DOF robot
    
    # Disable mouse picking and enable camera control
    #p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)  # Disable object manipulation
    #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Enable GUI controls
    
    # Set initial camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Visualize test points and compute distances
    for point in test_points:
        # Add visual marker for test point
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=point
        )
        
        # Compute and print distance
        distance = compute_point_to_robot_distance(robotId, point, home_angles, visualize=True)
        print(f"Distance to point {point}: {distance:.3f} meters")
    
    print("\nTest points visualized as red spheres.")
    print("Press Ctrl+C to exit.")
    
    while True:
        p.stepSimulation()
        time.sleep(0.1)

if __name__ == "__main__":
    test_distance_computation()
