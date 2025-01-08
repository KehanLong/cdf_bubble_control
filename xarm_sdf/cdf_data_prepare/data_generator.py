import torch
import os
import numpy as np
import sys
from typing import Dict, Tuple, Optional
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
        self.workspace = [[-0.6, -0.6, 0.0],  # min x,y,z
                         [0.6, 0.6, 1.0]]      # max x,y,z
        self.n_discrete = 20       # Points per dimension for workspace discretization
        self.batchsize = 1000      # Batch size for configuration sampling
        self.epsilon = 1e-3         # Distance threshold for valid configurations
        
        self.visualizer = SDFVisualizer(device)
        
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
        
        # Ensure we're using torch operations that maintain gradients
        sdf_values = self.robot_sdf.query_sdf(
            points=points,
            joint_angles=q,
            return_gradients=False
        )
        
        # Use torch.min instead of accessing values directly to maintain gradient
        d, min_indices = torch.min(sdf_values, dim=1)
        
        if return_index:
            return d, min_indices
        return d, None

    def wrap_angles(self, q: torch.Tensor) -> torch.Tensor:
        """
        Wrap angles to [-π, π] range
        """
        return torch.atan2(torch.sin(q), torch.cos(q))

    def find_valid_configurations(self, 
                                x: torch.Tensor,
                                initial_q: Optional[torch.Tensor] = None,
                                batchsize: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find valid configurations that place the robot at the given point using batch optimization
        """
        if batchsize is None:
            batchsize = self.batchsize  # Use class default (10000)
        
        # Initialize a large batch of random configurations
        q = torch.rand(batchsize, 6, device=self.device) * (self.q_max - self.q_min) + self.q_min
        q = q.detach().requires_grad_(True)
        
        def cost_function(q: torch.Tensor) -> torch.Tensor:
            q_wrapped = self.wrap_angles(q)
            q_reshaped = q_wrapped.reshape(-1, 6)
            d, _ = self.compute_sdf(x.unsqueeze(0), q_reshaped)
            cost = (d**2).sum()
            return cost

        try:
            result = minimize(
                cost_function,
                q,
                method='l-bfgs',
                options={
                    'max_iter': 50,
                    'line_search': 'strong-wolfe',
                    'disp': False
                }
            )
            
            # Get final distances and link indices
            with torch.no_grad():
                q_result = result.x.reshape(-1, 6)
                distances, link_indices = self.compute_sdf(x.unsqueeze(0), q_result, return_index=True)
                
                # Filter valid configurations
                valid_mask = (torch.abs(distances) < self.epsilon) & \
                            ((q_result > self.q_min) & (q_result < self.q_max)).all(dim=1)
                
                valid_q = q_result[valid_mask]
                valid_indices = link_indices[valid_mask]
                
                print(f'Found {len(valid_q)} valid configurations out of {batchsize} attempts '
                      f'for point {x.cpu().numpy()}')
                
                return valid_q, valid_indices
                
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return torch.zeros(0, 6, device=self.device), torch.zeros(0, device=self.device)

    def generate_dataset(self, save_path: str = None) -> Dict:
        """
        Generate CDF training dataset
        """
        print("\n=== Starting Dataset Generation ===")
        
        # 1. Create grid of points in workspace
        x = torch.linspace(self.workspace[0][0], self.workspace[1][0], self.n_discrete, device=self.device)
        y = torch.linspace(self.workspace[0][1], self.workspace[1][1], self.n_discrete, device=self.device)
        z = torch.linspace(self.workspace[0][2], self.workspace[1][2], self.n_discrete, device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        
        print(f"\nTotal grid points to process: {len(points)} ({self.n_discrete}×{self.n_discrete}×{self.n_discrete})")
        
        # 2. Find zero configurations for each point
        zero_configs = {}
        total_zero_configs = 0
        start_time = time.time()
        
        for i, point in enumerate(points):
            valid_q, _ = self.find_valid_configurations(point)
            if len(valid_q) > 0:
                zero_configs[i] = valid_q
                total_zero_configs += len(valid_q)
                
            if (i + 1) % 1 == 0 or i == len(points) - 1:
                elapsed_time = time.time() - start_time
                avg_time_per_point = elapsed_time / (i + 1)
                remaining_points = len(points) - (i + 1)
                estimated_remaining_time = remaining_points * avg_time_per_point
                
                print(f'\nProgress: {i+1}/{len(points)} points ({(i+1)/len(points)*100:.1f}%)')
                print(f'Zero configs found so far: {total_zero_configs}')
                print(f'Average configs per point: {total_zero_configs/(i+1):.2f}')
                print(f'Time elapsed: {elapsed_time/60:.1f} minutes')
                print(f'Estimated remaining time: {estimated_remaining_time/60:.1f} minutes')
                print(f'Points with valid configs: {len(zero_configs)}/{i+1}')

        print("\n=== Starting Training Pair Generation ===")
        
        # 3. Generate random configurations for training
        n_samples = 200
        random_configs = torch.rand(n_samples, 6, device=self.device) * (self.q_max - self.q_min) + self.q_min
        print(f"\nGenerating {n_samples} random configurations")

        # 4. Generate training pairs
        training_data = {
            'joint_angles': [],
            'points': [],
            'cdf_values': []
        }
        
        total_pairs = 0
        pair_start_time = time.time()

        for i, q in enumerate(random_configs):
            for j, point in enumerate(points):
                if j in zero_configs:
                    # Compute distance to all zero configurations for this point
                    zero_q = zero_configs[j]
                    distances = torch.norm(zero_q - q.unsqueeze(0), dim=1)
                    min_distance = torch.min(distances)
                    
                    training_data['joint_angles'].append(q.cpu().numpy())
                    training_data['points'].append(point.cpu().numpy())
                    training_data['cdf_values'].append(min_distance.cpu().numpy())
                    total_pairs += 1
            
            if (i + 1) % 10 == 0 or i == len(random_configs) - 1:
                elapsed = time.time() - pair_start_time
                print(f'\nProcessed {i+1}/{len(random_configs)} configurations')
                print(f'Training pairs generated: {total_pairs}')
                print(f'Average pairs per config: {total_pairs/(i+1):.1f}')
                print(f'Time elapsed: {elapsed/60:.1f} minutes')

        # Convert to numpy arrays
        print("\n=== Finalizing Dataset ===")
        for key in training_data:
            training_data[key] = np.array(training_data[key])
        
        print(f"\nFinal dataset statistics:")
        print(f"Total training pairs: {len(training_data['joint_angles'])}")
        print(f"Total points with valid configs: {len(zero_configs)}")
        print(f"Total zero configurations: {total_zero_configs}")
        print(f"Average zero configs per valid point: {total_zero_configs/len(zero_configs):.2f}")

        if save_path:
            np.save(save_path, training_data)
            print(f"\nDataset saved to: {save_path}")
            
        return training_data

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
    
    # Test visualization with a few points
    # generator.visualize_test_points(n_points=3)
    
    # Generate and save full dataset
    data = generator.generate_dataset(os.path.join(generator.save_dir, 'xarm_cdf_data.npy')) 