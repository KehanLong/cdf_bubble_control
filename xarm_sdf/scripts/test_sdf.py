import sys
import os

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
sys.path.append(project_root)

from xarm_sdf.models.xarm_model import XArmFK
import torch
import numpy as np

def test_fk():
    # Create instance of XArmFK
    xarm = XArmFK()
    
    # Test with zero configuration
    q_zero = torch.zeros(6)
    transforms = xarm.fkine(q_zero)
    
    # Print end-effector position
    print("End-effector position at zero config:")
    print(transforms[-1][0, :3, 3])  # Last transform, translation component
    
    # Test joint limits
    q_valid = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_invalid = torch.tensor([4*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("\nValid joint configuration:", xarm.check_joint_limits(q_valid))
    print("Invalid joint configuration:", xarm.check_joint_limits(q_invalid))

if __name__ == "__main__":
    test_fk()