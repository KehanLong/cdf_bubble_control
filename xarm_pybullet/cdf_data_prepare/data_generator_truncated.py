import torch
import os
import numpy as np
import sys
from typing import Dict, Tuple, Optional
import trimesh
import cProfile
import pstats
from pstats import SortKey

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
        self.epsilon = 1e-2         # Distance threshold for valid configurations
        
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
                                batchsize: Optional[int] = None,
                                max_valid: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find valid configurations that place the robot at the given point using batch optimization
        
        Args:
            x: Target point in workspace
            initial_q: Initial configuration guess
            batchsize: Size of optimization batch
            max_valid: Maximum number of valid configurations to find
        """
        if batchsize is None:
            batchsize = self.batchsize
            
        valid_q_list = []
        valid_indices_list = []
        n_attempts = 0
        max_attempts = 30  # Reduced from 50
        
        # Pre-compute initial configurations for all attempts
        all_q = []
        if initial_q is not None:
            noise_scale = 0.3
            for _ in range(max_attempts):
                noise = torch.randn(batchsize - 1, 6, device=self.device) * noise_scale
                q_random = initial_q.unsqueeze(0) + noise
                q_random = torch.clamp(q_random, self.q_min, self.q_max)
                q = torch.cat([initial_q.unsqueeze(0), q_random], dim=0)
                all_q.append(q)
        else:
            for _ in range(max_attempts):
                q = torch.rand(batchsize, 6, device=self.device) * (self.q_max - self.q_min) + self.q_min
                all_q.append(q)
        
        while len(valid_q_list) < max_valid and n_attempts < max_attempts:
            q = all_q[n_attempts]
            q = q.detach().requires_grad_(True)
            
            try:
                # Reduced max iterations and increased tolerance
                result = minimize(
                    lambda q: self._cost_function(q, x),
                    q,
                    method='l-bfgs',
                    tol=1e-1,
                    options={'max_iter': 10},  # Reduced from 20
                    disp=0
                )
                
                # Process results
                with torch.no_grad():
                    q_result = result.x.reshape(-1, 6)
                    distances, link_indices = self.compute_sdf(x.unsqueeze(0), q_result, return_index=True)
                    
                    # Filter valid configurations
                    valid_mask = (torch.abs(distances) < self.epsilon) & \
                               ((q_result > self.q_min) & (q_result < self.q_max)).all(dim=1)
                    
                    new_valid_q = q_result[valid_mask]
                    new_valid_indices = link_indices[valid_mask]
                    
                    if len(new_valid_q) > 0:
                        valid_q_list.append(new_valid_q)
                        valid_indices_list.append(new_valid_indices)
                        
                        # Early stopping if we found enough configurations
                        total_valid = sum(len(q) for q in valid_q_list)
                        if total_valid >= max_valid:
                            break
                    
            except Exception as e:
                print(f"Optimization failed: {str(e)}")
            
            n_attempts += 1
        
        # Combine and limit results
        if valid_q_list:
            valid_q = torch.cat(valid_q_list, dim=0)
            valid_indices = torch.cat(valid_indices_list, dim=0)
            
            if len(valid_q) > max_valid:
                valid_q = valid_q[:max_valid]
                valid_indices = valid_indices[:max_valid]
            
            return valid_q, valid_indices
        
        return torch.zeros(0, 6, device=self.device), torch.zeros(0, device=self.device)

    def _cost_function(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Helper function for optimization cost"""
        q_wrapped = self.wrap_angles(q)
        q_reshaped = q_wrapped.reshape(-1, 6)
        d, _ = self.compute_sdf(x.unsqueeze(0), q_reshaped)
        return (d**2).sum()

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

    def test_reverse_sampling(self, n_configs: int = 1, points_per_level: int = 100, cdf_threshold: float = 0.8):
        """
        Test reverse sampling approach with filtering for invalid CDF values
        """
        print("\n=== Starting Data Generation ===")
        print(f"Total configurations to process: {n_configs}")
        
        # Generate random configurations
        configs = torch.rand(n_configs, 6, device=self.device) * (self.q_max - self.q_min) + self.q_min
        
        # Create dense grid for initial sampling
        x = torch.linspace(self.workspace[0][0], self.workspace[1][0], 80, device=self.device)
        y = torch.linspace(self.workspace[0][1], self.workspace[1][1], 80, device=self.device)
        z = torch.linspace(self.workspace[0][2], self.workspace[1][2], 80, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        
        # Store valid training pairs
        training_pairs = []
        total_valid_pairs = 0  # Only counting non-zero level set pairs
        start_time = time.time()
        
        for i, q in enumerate(configs):
            print(f"\nProcessing configuration {i+1}/{n_configs}")
            
            # 1. Find SDF values for all grid points
            with torch.no_grad():
                points_expanded = grid_points.unsqueeze(0)
                q_expanded = q.unsqueeze(0)
                sdf_values = self.robot_sdf.query_sdf(
                    points=points_expanded, 
                    joint_angles=q_expanded, 
                    return_gradients=False
                )
                sdf_values = sdf_values.squeeze(0)
            
            # 2. Sample points at different levels
            levels = [0.0, 0.04, 0.08, 0.12]
            sampled_points = []
            
            for level in levels:
                level_mask = torch.abs(sdf_values - level) < 0.002
                level_points = grid_points[level_mask]
                
                if len(level_points) > 0:
                    if level == 0.0:
                        selected_points = level_points
                    else:
                        if len(level_points) > points_per_level:
                            indices = torch.randperm(len(level_points))[:points_per_level]
                            selected_points = level_points[indices]
                        else:
                            selected_points = level_points
                    sampled_points.append((level, selected_points))
            
            # 3. Process sampled points and filter invalid CDF values
            valid_pairs = 0
            invalid_pairs = 0
            
            for level, points in sampled_points:
                # Handle zero level set points silently
                if abs(level) < 1e-3:
                    for point in points:
                        training_pairs.append({
                            'joint_angles': q.cpu().numpy(),
                            'point': point.cpu().numpy(),
                            'cdf_value': 0.0
                        })
                    continue
                
                # Process non-zero level sets with tracking
                for point in points:
                    valid_q, _ = self.find_valid_configurations(point, initial_q=q, batchsize=100)
                    
                    if len(valid_q) > 0:
                        distances = torch.norm(valid_q - q.unsqueeze(0), dim=1)
                        cdf_value = torch.min(distances).item()
                        
                        if cdf_value < cdf_threshold:
                            training_pairs.append({
                                'joint_angles': q.cpu().numpy(),
                                'point': point.cpu().numpy(),
                                'cdf_value': cdf_value
                            })
                            valid_pairs += 1
                        else:
                            invalid_pairs += 1
            
            total_valid_pairs += valid_pairs
            elapsed_time = time.time() - start_time
            avg_time_per_config = elapsed_time / (i + 1)
            remaining_configs = n_configs - (i + 1)
            estimated_remaining_time = remaining_configs * avg_time_per_config
            
            print(f"Non-zero level valid pairs in this config: {valid_pairs} (Invalid: {invalid_pairs})")
            print(f"Total non-zero level valid pairs so far: {total_valid_pairs}")
            print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
            print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
        
        print(f"\n=== Data Generation Complete ===")
        print(f"Total configurations processed: {n_configs}")
        print(f"Total training pairs collected: {len(training_pairs)}")
        print(f"Total non-zero level valid pairs: {total_valid_pairs}")
        
        return training_pairs

    def profile_reverse_sampling(self, n_configs: int = 1, points_per_level: int = 100):
        """
        Profiled version of test_reverse_sampling
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        self.test_reverse_sampling(n_configs, points_per_level)
        
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(30)  # Print top 30 time-consuming functions

if __name__ == "__main__":
    generator = XArmCDFDataGenerator()
    
    # Generate data with 200 configurations
    print("Starting data generation with 200 configurations...")
    training_data = generator.test_reverse_sampling(n_configs=100, points_per_level=10, cdf_threshold=0.8)
    
    # Save the data
    save_path = os.path.join(generator.save_dir, 'cdf_training_data.npy')
    np.save(save_path, training_data)
    print(f"\nData saved to: {save_path}")

