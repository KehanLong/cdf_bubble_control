import pybullet as p
import pybullet_data
import numpy as np
import torch
import os
import sys
from time import sleep

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from xarm_sdf.models.xarm_model import XArmFK

def debug_fk_discrepancy():
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load robot
    urdf_path = os.path.join(project_root, "xarm_sdf", "xarm_description", "xarm6_robot.urdf")
    robotId = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    
    # Initialize our FK model
    robot_model = XArmFK()
    
    # Create visual markers for different FK results
    our_fk_marker = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[1, 0, 0, 1]  # Red for our FK
    )
    
    pybullet_fk_marker = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=[0, 1, 0, 1]  # Green for PyBullet FK
    )
    
    # Test configurations
    test_configs = [
        [0, 0, 0, 0, 0, 0],  # Home position
        [0.5, 0.5, 0.0, 0.0, 0.5, 0.1],
        [1.0, -1.0, 1.1, -0.3, -0.2, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.5],
    ]
    
    for i, joint_angles in enumerate(test_configs):
        print(f"\nTesting Configuration {i + 1}:")
        print(f"Joint angles: {[f'{angle:.3f}' for angle in joint_angles]}")
        
        # Set robot configuration
        for j in range(6):
            p.resetJointState(robotId, j+1, joint_angles[j])
        
        # Get PyBullet FK
        link_state = p.getLinkState(robotId, 6)
        pybullet_pos = link_state[0]
        
        # Get our FK
        config = torch.tensor([joint_angles], device='cuda')
        our_pos = robot_model.fkine(config).squeeze()[-1, :].cpu().numpy()
        
        # Create visual markers
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=our_fk_marker,
            basePosition=our_pos.tolist(),
        )
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pybullet_fk_marker,
            basePosition=pybullet_pos,
        )
        
        print("Our FK position:", our_pos)
        print("PyBullet position:", pybullet_pos)
        print("Difference:", our_pos - np.array(pybullet_pos))
        
        # Print joint info for debugging
        for j in range(p.getNumJoints(robotId)):
            joint_info = p.getJointInfo(robotId, j)
            print(f"Joint {j}: {joint_info[1]}, type: {joint_info[2]}")
        
        input("Press Enter for next configuration...")
        p.removeAllUserDebugItems()
    
    p.disconnect()

if __name__ == "__main__":
    debug_fk_discrepancy() 