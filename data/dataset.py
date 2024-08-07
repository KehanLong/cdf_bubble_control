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