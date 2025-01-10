import torch
import os
import numpy as np
import sys
from typing import Dict, Tuple, Optional, List
import trimesh

# Add parent directory to path to import xarm modules
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, '..'))

from models.xarm_model import XArmFK
from robot_sdf import RobotSDF
from torchmin import minimize
import time
import math
from utils.visualization import SDFVisualizer
import matplotlib.pyplot as plt

class XArmCDFDataGenerator:
    def __init__(self, device: str = 'cuda'):
        # Initialize robot models
        self.robot_fk = XArmFK(device=device)
        self.robot_sdf = RobotSDF(device=device)
        
        # Store joint limits
        self.q_max = self.robot_fk.joint_limits[:, 1]
        self.q_min = self.robot_fk.joint_limits[:, 0]
        
        # Device
        self.device = device
        
        # Data generation parameters
        self.workspace = [[-0.5, -0.5, 0.0],  # min x,y,z
                         [0.5, 0.5, 1.0]]      # max x,y,z
        self.n_discrete = 40      # Points per dimension for workspace discretization
        self.batchsize = 30000    # Batch size for configuration sampling
        self.epsilon = 5e-3       # Distance threshold to filter data
        
        # Update save directory
        self.save_dir = os.path.join(CUR_DIR, '..', 'data', 'cdf_data')
        os.makedirs(self.save_dir, exist_ok=True)

    def compute_sdf(self, 
                   x: torch.Tensor, 
                   q: torch.Tensor, 
                   return_index: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute SDF values for given points and configurations
        
        Args:
            x: Points in workspace (Nx, 3)
            q: Joint configurations (Nq, 6)
            return_index: Whether to return closest link indices
            
        Returns:
            d: SDF values (Nq,)
            idx: (Optional) Closest link indices (Nq,)
        """
        points = x.unsqueeze(0).expand(len(q), -1, -1)  # (Nq, Nx, 3)
        
        # Get both SDF values and link indices from query_sdf_batch
        sdf_values, link_indices = self.robot_sdf.query_sdf_batch(
            points=points,
            joint_angles=q,
            return_link_id=True,  # Make sure to request link IDs
            return_gradients=False
        )
        
        # Debug prints
        # if return_index:
        #     print("\nSDF Debug:")
        #     print(f"SDF values shape: {sdf_values.shape}")
        #     print(f"Link indices shape: {link_indices.shape}")
        #     print(f"Unique link indices: {torch.unique(link_indices).cpu().numpy()}")
        #     print(f"Link indices distribution: {torch.bincount(link_indices.flatten()).cpu().numpy()}")
        
        if return_index:
            return sdf_values, link_indices
        return sdf_values, None

    def wrap_angles(self, q: torch.Tensor) -> torch.Tensor:
        """
        Wrap angles to [-π, π] range
        """
        return torch.atan2(torch.sin(q), torch.cos(q))

    def _cost_function(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Cost function for optimization
        Args:
            q: Joint configurations (N, 6)
            x: Target point (3,)
        Returns:
            cost: Scalar cost value
        """
        q_wrapped = self.wrap_angles(q)
        q_reshaped = q_wrapped.reshape(-1, 6)
        d, _ = self.compute_sdf(x.unsqueeze(0), q_reshaped)
        cost = (d**2).sum()
        return cost

    def find_valid_configurations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find valid configurations using single large batch optimization
        """
        t0 = time.time()
        
        # Initialize random configurations
        q = torch.rand(self.batchsize, 6, device=self.device) * (self.q_max - self.q_min) + self.q_min
        q = q.detach().requires_grad_(True)
        
        try:
            result = minimize(
                lambda q: self._cost_function(q, x),
                q,
                method='l-bfgs', 
                options=dict(line_search='strong-wolfe'),
                max_iter=50,
                disp=0
            )
            
            # Get final distances and link indices
            with torch.no_grad():
                q_result = result.x.reshape(-1, 6)
                distances, link_indices = self.compute_sdf(x.unsqueeze(0), q_result, return_index=True)
                
                # Filter valid configurations
                mask = torch.abs(distances) < self.epsilon
                boundary_mask = ((q_result > self.q_min) & (q_result < self.q_max)).all(dim=1)
                final_mask = mask.squeeze() & boundary_mask  # Add squeeze() to make mask 1D
                
                valid_q = q_result[final_mask]
                valid_indices = link_indices[final_mask]
                
                # Debug prints after filtering
                if len(valid_indices) > 0:
                    print(f"Unique valid link indices: {torch.unique(valid_indices).cpu().numpy()}")
                    print(f"Valid link indices distribution: {torch.bincount(valid_indices.flatten()).cpu().numpy()}")
                
                print(f'number of q_valid: \t{len(valid_q)} \t time cost:{time.time()-t0:.2f}')
                
                return valid_q, valid_indices
                
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return torch.zeros(0, 6, device=self.device), torch.zeros(0, device=self.device)

    def save_contact_database(self, data: Dict, save_path: str):
        """Save contact database in memory-efficient compressed format"""
        compressed_db = {
            'points': np.array([data[i]['x'] for i in data.keys()]),  # [N, 3]
            'contact_configs': [data[i]['q'] for i in data.keys()],     # List of [M_i, 6] arrays
            'link_indices': [data[i]['idx'] for i in data.keys()]      # List of [M_i] arrays
        }
        np.save(save_path, compressed_db)
        print(f"\nSaved contact database to: {save_path}")
        print(f"Number of points: {len(compressed_db['points'])}")
        print(f"Average configs per point: {np.mean([len(configs) for configs in compressed_db['contact_configs']]):.1f}")

    def generate_dataset(self, save_path: str = None) -> Dict:
        """
        Generate CDF training dataset
        """
        print("\n=== Starting Contact Database Generation ===")
        
        # Create grid of points in workspace
        x = torch.linspace(self.workspace[0][0], self.workspace[1][0], self.n_discrete, device=self.device)
        y = torch.linspace(self.workspace[0][1], self.workspace[1][1], self.n_discrete, device=self.device)
        z = torch.linspace(self.workspace[0][2], self.workspace[1][2], self.n_discrete, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        
        print(f"\nTotal grid points to process: {len(points)} ({self.n_discrete}×{self.n_discrete}×{self.n_discrete})")
        
        # Process each point
        data = {}
        start_time = time.time()
        
        for i, point in enumerate(points):
            print(f"\nProcessing point {i+1}/{len(points)}: {point.cpu().numpy()}")
            
            q, idx = self.find_valid_configurations(point)
            
            data[i] = {
                'x': point.cpu().numpy(),
                'q': q.cpu().numpy(),
                'idx': idx.cpu().numpy(),
            }
            
            # Print progress
            elapsed_time = time.time() - start_time
            avg_time_per_point = elapsed_time / (i + 1)
            remaining_points = len(points) - (i + 1)
            estimated_remaining_time = remaining_points * avg_time_per_point
            
            print(f'Progress: {i+1}/{len(points)} points ({(i+1)/len(points)*100:.1f}%)')
            print(f'Time elapsed: {elapsed_time/60:.1f} minutes')
            print(f'Estimated remaining time: {estimated_remaining_time/60:.1f} minutes')

            # Save periodically
            # if save_path and i % 100 == 0 and i > 0:
            #     temp_save_path = save_path.replace('.npy', f'_checkpoint_{i}.npy')
            #     self.save_contact_database(data, temp_save_path)
            #     print(f"\nCheckpoint saved to: {temp_save_path}")

        # Final save
        if save_path:
            self.save_contact_database(data, save_path)
        
        return data

    def visualize_test_points(self, n_points: int = 1):
        """
        Visualize test points and their valid configurations
        """
        # Test with just one point first
        point = torch.tensor([0.2, 0.5, 0.4], device=self.device)  # A reasonable point in workspace
        
        print(f"\nTesting with point: {point.cpu().numpy()}")
        valid_q, link_indices = self.find_valid_configurations(point)
        
        if len(valid_q) > 0:
            print(f"\nFound valid configuration:")
            print(f"Configuration: {valid_q[0].cpu().numpy()}")
            print(f"Link index: {link_indices[0].cpu().numpy()}")
            
            # Create visualization
            scene = self.visualizer.visualize_sdf(valid_q[0])
            
            # Add sphere at target point
            sphere = trimesh.primitives.Sphere(radius=0.02, center=point.cpu().numpy())
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red color
            scene.add_geometry(sphere)
            
            # Show scene
            scene.show()
        else:
            print("No valid configurations found")

    def test_differentiability(self):
        # Create test point and configuration
        x = torch.tensor([0.5, 0.3, 0.3], device=self.device)
        q = torch.zeros(1, 6, device=self.device, requires_grad=True)
        
        # Compute SDF
        d, _ = self.compute_sdf(x.unsqueeze(0), q)
        
        print(f"SDF value: {d}")
        
        try:
            # Try to compute gradient
            loss = d.sum()
            loss.backward()
            print("\nGradient check:")
            print(f"Gradient shape: {q.grad.shape}")
            print(f"Gradient values: {q.grad.cpu().numpy()}")
            print(f"Are there any NaN in gradient?: {torch.isnan(q.grad).any()}")
            print(f"Are there any Inf in gradient?: {torch.isinf(q.grad).any()}")
            print("\nSDF is differentiable!")
        except Exception as e:
            print("\nSDF is not differentiable:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")

if __name__ == "__main__":
    generator = XArmCDFDataGenerator()
    
    # Generate and save contact database
    data = generator.generate_dataset(os.path.join(generator.save_dir, 'bfgs_contact_db.npy')) 