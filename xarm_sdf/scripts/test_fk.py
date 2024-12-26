import pybullet as p
import pybullet_data
import numpy as np
import os
import sys
import torch


# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def test_fk_with_pybullet(robot_type="panda"):
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Set up robot-specific parameters
    if robot_type == "panda":
        from xarm_sdf.models.panda_model import PandaFK
        urdf_path = os.path.join(project_root, "xarm_sdf", "xarm_description", "panda_urdf", "panda.urdf")
        robot_model = PandaFK()
        num_joints = 7
        config = torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
    else:  # xarm
        from xarm_sdf.models.xarm_model import XArmFK
        urdf_path = os.path.join(project_root, "xarm_sdf", "xarm_description", "xarm6_robot.urdf")
        robot_model = XArmFK()
        num_joints = 6
        config = torch.tensor([[0, 0, 0, 0, 0, 0]])
    
    robotId = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
    fk_positions = robot_model.fkine(config)
    
    # Create a small sphere visual shape for FK points
    sphere_radius = 0.02
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[0, 0, 1, 1]  # Blue color
    )
    
    print(f"\nComparing Joint Positions for {robot_type.upper()}:")
    print("=" * 50)
    print(f"Number of FK points: {fk_positions.shape[1] if isinstance(fk_positions, torch.Tensor) else len(fk_positions)}")
    
    # Get PyBullet joint positions for comparison
    print("\nPyBullet Joint Positions:")
    for i in range(num_joints):
        link_state = p.getLinkState(robotId, i+1)
        joint_pos = link_state[4]  # World position of joint
        print(f"Joint {i+1}:")
        print(f"PyBullet Position: {[f'{x:.6f}' for x in joint_pos]}")
    
    # Visualize FK positions
    print("\nFK Positions:")
    if isinstance(fk_positions, torch.Tensor):
        # For Panda model
        for i in range(fk_positions.shape[1]):
            our_pos = fk_positions[0, i].detach().numpy()
            print(f"FK Point {i}: {[f'{x:.6f}' for x in our_pos]}")
            
            # Visualize FK result
            translated_pos = our_pos.copy()
            translated_pos[0] += 0.5
            
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sphere_visual,
                basePosition=translated_pos.tolist(),
            )
            p.addUserDebugLine(
                our_pos.tolist(),
                translated_pos.tolist(),
                [0,0,1]
            )
    else:
        # For xArm model
        for i, transform in enumerate(fk_positions):
            our_pos = transform[:3, 3].detach().numpy()
            print(f"FK Point {i}: {[f'{x:.6f}' for x in our_pos]}")
            
            translated_pos = our_pos.copy()
            translated_pos[0] += 0.5
            
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sphere_visual,
                basePosition=translated_pos.tolist(),
            )
            p.addUserDebugLine(
                our_pos.tolist(),
                translated_pos.tolist(),
                [0,0,1]
            )
    
    input("Press Enter to exit...")
    p.disconnect()

if __name__ == "__main__":
    # Test both robots
    #test_fk_with_pybullet("panda")
    test_fk_with_pybullet("xarm")