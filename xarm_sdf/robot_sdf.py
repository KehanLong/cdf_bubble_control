import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
import numpy as np
from training.network import SDFNetwork
from models.xarm6_differentiable_fk import fk_xarm6_torch

class RobotSDF:
    def __init__(self, device='cuda', with_gripper=True):
        self.device = device
        self.with_gripper = with_gripper
        # Start from link 1 (skip base link 0)
        self.num_links = 7 if with_gripper else 6
        
        # Load all trained SDF models
        self.models = self._load_sdf_models()
        
        # Load scaling parameters
        data_dir = project_root / "data" / "sdf_data"
        self.offsets = []
        self.scales = []
        # Loading links (6 movable arm links + optional gripper), starting from link 1
        for i in range(1, self.num_links + 1):  # Note: i now starts from 1
            if i < 7:
                data = np.load(data_dir / f"link_{i}_link{i}.npz")
            else:
                data = np.load(data_dir / f"link_{i}_gripper.npz")
            # Explicitly convert to float32
            self.offsets.append(torch.from_numpy(data['original_center']).float().to(device))
            self.scales.append(torch.tensor(data['original_scale'], dtype=torch.float32).to(device))
        
        self.offsets = torch.stack(self.offsets)  # [num_links, 3]
        self.scales = torch.stack(self.scales)     # [num_links]
    
    def _load_sdf_models(self):
        """Load trained models for all links including optional gripper"""
        models = []
        model_dir = project_root / "trained_models"
        # Loading models (6 movable arm links + optional gripper), starting from link 1
        for i in range(1, self.num_links + 1):  # Note: i now starts from 1
            model_path = model_dir / f"link_{i}" / "best_model.pt"
            checkpoint = torch.load(model_path)
            
            model = SDFNetwork().to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        return models
    
    def query_sdf(self, points, joint_angles, return_link_id=False, return_gradients=False, gradient_method='analytic'):
        """
        Query SDF for points given joint angles
        Args:
            gradient_method: 'finite_diff', 'analytic', or 'both'
        """
        points = points.to(dtype=torch.float32)
        joint_angles = joint_angles.to(dtype=torch.float32)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        if return_gradients and (gradient_method == 'analytic' or gradient_method == 'both'):
            joint_angles.requires_grad_(True)
        
        # Get transforms for each link (now only for movable links)
        transforms_list = []
        for b in range(B):
            # Get transforms from FK function
            link_transforms = fk_xarm6_torch(joint_angles[b], with_gripper=self.with_gripper)
            
            # If with_gripper, use Link 6's transform for the gripper (skip the last transform)
            if self.with_gripper:
                # Use all transforms except the last one, and repeat Link 6's transform
                transforms = link_transforms[:-1] + [link_transforms[-2]]
            else:
                transforms = link_transforms
                
            transforms_list.append(transforms)
        
        transforms = torch.stack([torch.stack(t) for t in transforms_list])
        
        # Transform points to each link's local frame (no need for base transform)
        points_expanded = points.unsqueeze(1).expand(-1, self.num_links, -1, -1)
        inv_transforms = torch.linalg.inv(transforms)
        local_points = torch.einsum('bkij,bknj->bkni', inv_transforms, 
                                  torch.cat([points_expanded, 
                                           torch.ones_like(points_expanded[...,:1])], dim=-1))
        local_points = local_points[...,:3]
        
        # Scale points
        scaled_points = ((local_points - self.offsets.view(1, self.num_links, 1, 3)) / 
                        self.scales.view(1, self.num_links, 1, 1)).to(dtype=torch.float32)
        
        # Query each link's SDF
        sdf_values = []
        for i, model in enumerate(self.models):
            points_i = scaled_points[:,i].reshape(B*N, 3)
            with torch.set_grad_enabled(True):
                sdf = model(points_i).reshape(B, N, 1)
            sdf_values.append(sdf)
        
        sdf_values = torch.stack(sdf_values, dim=1)  # [B, num_links, N, 1]
        sdf_values = sdf_values.squeeze(-1) * self.scales.view(1, self.num_links, 1)
        
        # Get minimum distance and corresponding link
        min_sdf, link_ids = torch.min(sdf_values, dim=1)  # [B, N]
        
        result = [min_sdf]
        
        if return_link_id:
            result.append(link_ids)
        
        if return_gradients:
            if gradient_method == 'analytic':
                #print("\nComputing analytic gradients...")
                analytic_grads = torch.zeros((B, N, 6), device=self.device)
                for b in range(B):
                    for i in range(N):
                        try:
                            grad = torch.autograd.grad(min_sdf[b, i], 
                                                     joint_angles,
                                                     retain_graph=True,
                                                     allow_unused=True)[0]
                            if grad is not None:
                                analytic_grads[b, i] = grad[b]
                        except Exception as e:
                            print(f"Warning: Gradient computation failed for batch {b}, point {i}: {e}")
                
                # print(f"Analytic gradients shape: {analytic_grads.shape}")
                # print(f"Analytic gradients mean: {analytic_grads.mean():.6f}")
                # print(f"Analytic gradients std: {analytic_grads.std():.6f}")
                result.append(analytic_grads)
                
            elif gradient_method == 'finite_diff':
                #print("\nComputing finite difference gradients...")
                delta = 0.001
                finite_diff_grads = []
                
                # For each joint
                for i in range(6):
                    perturbed_angles = joint_angles.clone()
                    perturbed_angles[:, i] += delta
                    perturbed_sdf = self.query_sdf(points, perturbed_angles)[0]
                    grad = (perturbed_sdf - min_sdf) / delta
                    finite_diff_grads.append(grad)
                
                finite_diff_grads = torch.stack(finite_diff_grads, dim=2)  # [B, N, 6]
                # print(f"Finite diff gradients shape: {finite_diff_grads.shape}")
                # print(f"Finite diff gradients mean: {finite_diff_grads.mean():.6f}")
                # print(f"Finite diff gradients std: {finite_diff_grads.std():.6f}")
                result.append(finite_diff_grads)
                
            elif gradient_method == 'both':
                # Compute both and return as dict
                analytic = self.query_sdf(points, joint_angles, return_link_id=False, 
                                        return_gradients=True, gradient_method='analytic')[1]
                finite = self.query_sdf(points, joint_angles, return_link_id=False, 
                                      return_gradients=True, gradient_method='finite_diff')[1]
                
                print("\nGradient comparison:")
                diff = analytic - finite
                print(f"Mean difference: {diff.abs().mean():.6f}")
                print(f"Max difference: {diff.abs().max():.6f}")
                print(f"Relative difference: {(diff.abs() / (analytic.abs() + 1e-6)).mean():.6f}")
                
                result.append({'analytic': analytic, 'finite_diff': finite})
        
        return tuple(result) if len(result) > 1 else result[0]
    
    def test_differentiability(self):
        # Create test inputs
        points = torch.randn(1, 3, 3, device=self.device)  # [batch, num_points, 3]
        joint_angles = torch.zeros(1, 6, device=self.device, requires_grad=True)
        
        # Forward pass
        sdf = self.query_sdf(points, joint_angles)
        
        # Check if gradients flow
        try:
            loss = sdf.sum()
            loss.backward()
            print("Gradient computation successful!")
            print(f"Joint angles gradient: {joint_angles.grad}")
        except Exception as e:
            print(f"Gradient computation failed: {str(e)}")

if __name__ == "__main__":
    device = 'cuda'
    robot_sdf = RobotSDF(device)
    
    # Fixed joint configuration
    joint_angles = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)  # [1, 6]
    print(f"\nTesting with joint angles: {joint_angles.cpu().numpy()[0]}")
    
    # Test points (in world frame)
    test_points = torch.tensor([
        # Point near base
        [0.0, 0.5, 0.1],
        # Point near middle (around link 2-3)
        [0.3, 0.2, 0.3],
        # Point near end-effector
        [0.3, 0.3, 0.5]
    ], device=device).unsqueeze(0)  # [1, 3, 3]
    
    print("\nTest points:")
    for i, p in enumerate(test_points[0]):
        print(f"Point {i}: {p.cpu().numpy()}")
    
    # Query SDF and gradients for these specific points
    sdf, link_ids, grads = robot_sdf.query_sdf(test_points, joint_angles, 
                                              return_link_id=True, 
                                              return_gradients=True,
                                              gradient_method='both')
    
    # Print results for each point
    print("\nResults:")
    for i in range(len(test_points[0])):
        print(f"\nPoint {i}:")
        print(f"SDF value: {sdf[0,i].item():.6f}")
        print(f"Closest link: {link_ids[0,i].item()}")
        print("Gradients:")
        print("  Analytic:", grads['analytic'][0,i].detach().cpu().numpy())
        print("  Finite diff:", grads['finite_diff'][0,i].detach().cpu().numpy())
        print(f"  Max difference: {(grads['analytic'][0,i] - grads['finite_diff'][0,i]).abs().max().item():.6f}")
    
    # Add this to your test code
    robot_sdf = RobotSDF(device='cuda')
    robot_sdf.test_differentiability()
    
    