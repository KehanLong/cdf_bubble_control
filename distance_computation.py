import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class RobotDistanceComputer:
    def __init__(self):
        # Start main GUI physics server
        self.gui_id = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)  # Disable object manipulation
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Start separate collision checking server
        self.col_id = p.connect(p.DIRECT)
        
        # Load robot in both servers
        self.robot_gui, self.robot_col = self._load_robots()
        
    def _load_robots(self):
        # Load robot in GUI server
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.gui_id)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "xarm", "xarm6_robot.urdf")
        
        robot_gui = p.loadURDF(urdf_path, useFixedBase=1, physicsClientId=self.gui_id)
        robot_col = p.loadURDF(urdf_path, useFixedBase=1, physicsClientId=self.col_id)
        
        return robot_gui, robot_col
    
    def compute_distance(self, point, joint_angles=None, visualize=False):
        """
        Compute distance from a point to the robot in a given configuration
        """
        # Set robot configuration in both servers
        if joint_angles is not None:
            for i, angle in enumerate(joint_angles):
                p.resetJointState(self.robot_gui, i, angle, physicsClientId=self.gui_id)
                p.resetJointState(self.robot_col, i, angle, physicsClientId=self.col_id)
        
        # Create temporary sphere for distance computation
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE, 
            radius=0.001, 
            physicsClientId=self.col_id
        )
        point_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            basePosition=point,
            physicsClientId=self.col_id
        )
        
        # Compute closest points
        closest_points = p.getClosestPoints(
            self.robot_col, 
            point_body, 
            distance=100.0,
            physicsClientId=self.col_id
        )
        
        min_distance = float('inf')
        if closest_points:
            min_distance = closest_points[0][8]
            if visualize:
                # Visualize in GUI server
                p.removeAllUserDebugItems(physicsClientId=self.gui_id)
                p.addUserDebugLine(
                    closest_points[0][5],  # pointA
                    closest_points[0][6],  # pointB
                    [0, 1, 0],  # color
                    2,  # line width
                    0,  # lifetime
                    physicsClientId=self.gui_id
                )
        
        # Cleanup
        p.removeBody(point_body, physicsClientId=self.col_id)
        p.removeCollisionShape(collision_shape, physicsClientId=self.col_id)
        
        return min_distance

def test_distance_computation():
    computer = RobotDistanceComputer()
    
    # Test points
    test_points = [
        [0.5, 0.0, 0.5],
        [0.3, 0.3, 0.3],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    
    # Test different configurations
    while True:
        # Random configuration
        q = np.pi * (np.random.random(6) - 0.5)
        
        for point in test_points:
            # Visualize test point
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=computer.gui_id
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=point,
                physicsClientId=computer.gui_id
            )
            
            # Compute and show distance
            dist = computer.compute_distance(point, q, visualize=True)
            print(f"Distance to point {point}: {dist:.3f}")
        
        input("Press Enter for next configuration...")
        p.removeAllUserDebugItems(physicsClientId=computer.gui_id)

if __name__ == "__main__":
    test_distance_computation()
