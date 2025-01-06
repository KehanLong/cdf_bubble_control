import torch
import math
import numpy as np


def rotx_torch(rx, device='cuda'):
    """Rotation about X by rx (radians), in PyTorch."""
    if not isinstance(rx, torch.Tensor):
        rx = torch.tensor(rx, dtype=torch.float32, device=device)
    c = torch.cos(rx)
    s = torch.sin(rx)
    R = torch.tensor([
        [1,  0,  0,  0],
        [0,  c, -s,  0],
        [0,  s,  c,  0],
        [0,  0,  0,  1]], dtype=rx.dtype, device=device)
    return R

def rotz_torch(rz, device='cuda'):
    """Rotation about Z by rz (radians), in PyTorch."""
    if not isinstance(rz, torch.Tensor):
        rz = torch.tensor(rz, dtype=torch.float32, device=device)
    c = torch.cos(rz)
    s = torch.sin(rz)
    R = torch.tensor([
        [ c, -s,  0,  0],
        [ s,  c,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]], dtype=rz.dtype, device=device)
    return R

def transl_torch(x, y, z, dtype=torch.float32, device='cuda'):
    """Translation by (x, y, z), as a 4x4 transform."""
    T = torch.eye(4, dtype=dtype, device=device)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

def fk_xarm6_torch(q, with_gripper=True):
    """
    Forward kinematics of xArm6 in PyTorch,
    matching the URDF exactly.
    Args:
        q: 1D tensor of shape [6].
        with_gripper: if True, include gripper transform
    returns: list of 4x4 transforms from base to each link (including gripper if with_gripper=True)
    """
    assert q.shape == (6,)
    device = q.device
    q1, q2, q3, q4, q5, q6 = q

    # Joint1: First fixed transform (translate & RPY), then joint rotation
    T01_fixed = transl_torch(0.0, 0.0, 0.267, dtype=q.dtype, device=device)
    T01 = T01_fixed @ rotz_torch(q1, device=device)

    # Joint2: First fixed transform (translate & RPY), then joint rotation
    T12_fixed = (
        transl_torch(0.0, 0.0, 0.0, dtype=q.dtype, device=device) @ 
        rotx_torch(-math.pi/2, device=device)
    )
    T12 = T12_fixed @ rotz_torch(q2, device=device)

    # Joint3: First fixed transform (translate & RPY), then joint rotation
    T23_fixed = transl_torch(0.0535, -0.2845, 0.0, dtype=q.dtype, device=device)
    T23 = T23_fixed @ rotz_torch(q3, device=device)

    # Joint4: First fixed transform (translate & RPY), then joint rotation
    T34_fixed = (
        transl_torch(0.0775, 0.3425, 0.0, dtype=q.dtype, device=device) @ 
        rotx_torch(-math.pi/2, device=device)
    )
    T34 = T34_fixed @ rotz_torch(q4, device=device)

    # Joint5: First fixed transform (translate & RPY), then joint rotation
    T45_fixed = (
        transl_torch(0.0, 0.0, 0.0, dtype=q.dtype, device=device) @ 
        rotx_torch(math.pi/2, device=device)
    )
    T45 = T45_fixed @ rotz_torch(q5, device=device)

    # Joint6: First fixed transform (translate & RPY), then joint rotation
    T56_fixed = (
        transl_torch(0.076, 0.097, 0.0, dtype=q.dtype, device=device) @ 
        rotx_torch(-math.pi/2, device=device)
    )
    T56 = T56_fixed @ rotz_torch(q6, device=device)

    # Multiply them all:
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56

    transforms = [T01, T02, T03, T04, T05, T06]
    
    if with_gripper:
        # Gripper transform relative to link6 (from URDF)
        # Add translation in z-direction for gripper length
        T6G_fixed = transl_torch(0.0, 0.0, 0.145, dtype=q.dtype, device=device)
        T0G = T06 @ T6G_fixed
        transforms.append(T0G)

    return transforms


def get_urdf_transform(xyz, rpy):
    """URDF transformation matrix"""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def compare_with_urdf(q=None):
    """Compare our FK transformations with URDF specifications"""
    if q is None:
        q = torch.zeros(6)  # Zero configuration
    
    
    # Define URDF transformations
    urdf_transforms = [
        {"xyz": [0, 0, 0.267], "rpy": [0, 0, 0]},         # Joint 1
        {"xyz": [0, 0, 0], "rpy": [-1.5708, 0, 0]},       # Joint 2
        {"xyz": [0.0535, -0.2845, 0], "rpy": [0, 0, 0]},  # Joint 3
        {"xyz": [0.0775, 0.3425, 0], "rpy": [-1.5708, 0, 0]}, # Joint 4
        {"xyz": [0, 0, 0], "rpy": [1.5708, 0, 0]},        # Joint 5
        {"xyz": [0.076, 0.097, 0], "rpy": [-1.5708, 0, 0]}  # Joint 6
    ]
    
    # Calculate URDF expected transforms
    T_urdf = np.eye(4)
    T_urdf_all = []
    
    for i, transform in enumerate(urdf_transforms):
        # First apply fixed transform from URDF
        T_fixed = get_urdf_transform(transform["xyz"], transform["rpy"])
        # Then apply joint rotation
        T_joint = get_urdf_transform([0,0,0], [0,0,q[i].item()])
        T_urdf = T_urdf @ T_fixed @ T_joint
        T_urdf_all.append(np.copy(T_urdf))
    
    # Get our FK transforms
    
    our_transforms = fk_xarm6_torch(q)
    
    # Convert to numpy for comparison
    our_transforms = [t.detach().numpy() if isinstance(t, torch.Tensor) else t for t in our_transforms]
    
    print("\nComparing transforms with URDF specifications:")
    print("============================================")
    
    for i in range(6):
        print(f"\nJoint {i+1}:")
        print(f"URDF transform:")
        print(np.round(T_urdf_all[i], 3))
        print(f"Our transform:")
        print(np.round(our_transforms[i], 3))
        print("Difference:")
        print(np.round(T_urdf_all[i] - our_transforms[i], 3))

if __name__ == "__main__":
    # Test zero configuration
    print("\nTesting zero configuration:")
    compare_with_urdf()
    



