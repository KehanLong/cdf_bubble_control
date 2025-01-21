import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cdf_training.network import CDFNetwork

class RobotCDF:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load trained CDF model
        self.model = self._load_cdf_model()
        
    def _load_cdf_model(self):
        """Load trained CDF model"""
        model_dir = project_root / "trained_models"
        model_path = model_dir / "best_model_gelu.pth"
        
        model = CDFNetwork(input_dims=8, output_dims=1, activation='gelu').to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model
    
    def query_cdf(self, points, joint_angles, return_gradients=False):
        """
        Query CDF for points given joint angles
        Args:
            points: [B, N, 2] tensor of 2D points
            joint_angles: [B, 2] tensor of joint angles
            return_gradients: bool, whether to return gradients w.r.t. joint angles
        Returns:
            cdf_values: [B, N] tensor of CDF values
            gradients: [B, N, 2] tensor of gradients (optional)
        """
        # print("\nDebug - query_cdf:")
        # print(f"Input points shape: {points.shape}")
        # print(f"Input joint_angles shape: {joint_angles.shape}")
        
        points = points.to(dtype=torch.float32, device=self.device)
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        if return_gradients:
            joint_angles = joint_angles.detach().clone()
            joint_angles.requires_grad_(True)
        
        # Remove extra dimensions from joint_angles first
        if len(joint_angles.shape) > 2:
            joint_angles = joint_angles.squeeze(1)
        
        # Prepare input features
        sin_q = torch.sin(joint_angles)  # [B, 2]
        cos_q = torch.cos(joint_angles)  # [B, 2]

        
        # Expand joint features for each point
        joint_features = torch.cat([
            joint_angles.unsqueeze(1).expand(-1, N, -1),  # [B, N, 2]
            sin_q.unsqueeze(1).expand(-1, N, -1),        # [B, N, 2]
            cos_q.unsqueeze(1).expand(-1, N, -1)         # [B, N, 2]
        ], dim=-1)  # [B, N, 6]

        
        # Concatenate with points
        inputs = torch.cat([joint_features, points], dim=-1)  # [B, N, 8]
        
        # Forward pass
        cdf_values = self.model(inputs.reshape(B*N, -1)).reshape(B, N)
        
        if return_gradients:
            gradients = torch.zeros((B, N, 2), device=self.device)
            
            for b in range(B):
                for n in range(N):
                    scalar_output = cdf_values[b, n:n+1]
                    grad_outputs = torch.ones_like(scalar_output)
                    
                    try:
                        grad = torch.autograd.grad(
                            scalar_output,
                            joint_angles,
                            grad_outputs=grad_outputs,
                            create_graph=True,
                            retain_graph=True
                        )[0]
                        
                        if grad is not None:
                            gradients[b, n] = grad[b]
                    except Exception as e:
                        print(f"Warning: Gradient computation failed for batch {b}, point {n}: {e}")
            
            return cdf_values, gradients
        
        return cdf_values

    def test_differentiability(self):
        """Test if the model is differentiable and check eikonal constraint"""
        # Create test inputs
        points = torch.randn(1, 3, 2, device=self.device)  # [batch, num_points, 2]
        joint_angles = torch.zeros(1, 2, device=self.device, requires_grad=True)
        
        # Get CDF values and gradients
        cdf_values, gradients = self.query_cdf(points, joint_angles, return_gradients=True)
        
        # Check gradient norms (eikonal constraint)
        gradient_norms = torch.norm(gradients, dim=-1)
        mean_norm = gradient_norms.mean().item()
        norm_deviation = (gradient_norms - 1.0).abs().mean().item()
        
        print("\nEikonal Constraint Check:")
        print(f"Mean gradient norm: {mean_norm:.4f} (should be close to 1.0)")
        print(f"Mean deviation from 1.0: {norm_deviation:.4f} (should be close to 0.0)")
        
        try:
            loss = cdf_values.sum()
            loss.backward()
            print("\nGradient computation successful!")
            print(f"Joint angles gradient: {joint_angles.grad}")
        except Exception as e:
            print(f"Gradient computation failed: {str(e)}")

if __name__ == "__main__":
    device = 'cuda'
    robot_cdf = RobotCDF(device)
    
    # Test points (2D)
    test_points = torch.tensor([
        [0.3, 0.2],
        [0.4, 0.3],
        [0.5, 0.4]
    ], device=device).unsqueeze(0)  # [1, 3, 2]
    
    # Test joint angles (2D)
    joint_angles = torch.tensor([[0.0, 0.0]], device=device)
    
    print("\nTesting CDF query...")
    print(f"Test points shape: {test_points.shape}")
    print(f"Joint angles shape: {joint_angles.shape}")
    
    # Query CDF values and gradients
    cdf_values, gradients = robot_cdf.query_cdf(test_points, joint_angles, return_gradients=True)
    
    print("\nResults:")
    for i in range(len(test_points[0])):
        print(f"\nPoint {test_points[0,i].cpu().numpy()}:")
        print(f"CDF value: {cdf_values[0,i].item():.6f}")
        print(f"Gradients: {gradients[0,i].detach().cpu().numpy()}")
    
    # Test differentiability
    robot_cdf.test_differentiability()
    
    # Add eikonal constraint visualization for test points
    print("\nEikonal Constraint at Test Points:")
    gradient_norms = torch.norm(gradients, dim=-1)
    for i in range(len(test_points[0])):
        print(f"\nPoint {test_points[0,i].cpu().numpy()}:")
        print(f"CDF value: {cdf_values[0,i].item():.6f}")
        print(f"Gradient norm: {gradient_norms[0,i].item():.6f} (should be close to 1.0)")
        print(f"Gradients: {gradients[0,i].detach().cpu().numpy()}") 