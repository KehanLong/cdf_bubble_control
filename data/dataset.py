import jax.numpy as jnp

class RobotArmDataset:
    def __init__(self, joint_angles, points, distances):
        self.joint_angles = jnp.array(joint_angles, dtype=jnp.float32)
        self.points = jnp.array(points, dtype=jnp.float32)
        self.distances = jnp.array(distances, dtype=jnp.float32)
        
        # Apply positional encoding to joint angles
        angles_sin = jnp.sin(self.joint_angles)
        angles_cos = jnp.cos(self.joint_angles)
        self.encoded_joint_angles = jnp.concatenate((self.joint_angles, angles_sin, angles_cos), axis=1)
    
    def __len__(self):
        return len(self.joint_angles)
    
    def __getitem__(self, idx):
        inputs = jnp.concatenate((self.encoded_joint_angles[idx], self.points[idx]))

        return inputs, self.distances[idx]

class SDFDataset:
    def __init__(self, points, distances):
        self.points = jnp.array(points, dtype=jnp.float32)
        self.distances = jnp.array(distances, dtype=jnp.float32)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        inputs = self.points[idx]
        return inputs, self.distances[idx]
    

import torch
import numpy as np

class RobotCDFDataset:
    def __init__(self, joint_angles, points, cdf_values):
        self.joint_angles = torch.tensor(np.array(joint_angles), dtype=torch.float32)
        self.points = torch.tensor(np.array(points), dtype=torch.float32)
        self.cdf_values = torch.tensor(np.array(cdf_values), dtype=torch.float32)
        
        # Apply positional encoding to joint angles
        angles_sin = torch.sin(self.joint_angles)
        angles_cos = torch.cos(self.joint_angles)
        self.encoded_joint_angles = torch.cat((self.joint_angles, angles_sin, angles_cos), dim=1)
    
    def __len__(self):
        return len(self.joint_angles) * len(self.points)
    
    def __getitem__(self, idx):
        config_idx = idx // len(self.points)
        point_idx = idx % len(self.points)
        inputs = torch.cat((self.encoded_joint_angles[config_idx], self.points[point_idx]))
        return inputs, self.cdf_values[config_idx, point_idx]