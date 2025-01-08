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
        model_dir = project_root / "trained_models" / "cdf"
        model_path = model_dir / "best_model.pth"
        
        model = CDFNetwork().to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model
    
    def query_cdf(self, points, joint_angles, return_gradients=False):
        """
        Query CDF for points given joint angles
        Args:
            points: [B, N, 3] tensor of 3D points
            joint_angles: [B, 6] tensor of joint angles
            return_gradients: bool, whether to return gradients w.r.t. joint angles
        Returns:
            cdf_values: [B, N] tensor of CDF values
            gradients: [B, N, 6] tensor of gradients (optional)
        """
        points = points.to(dtype=torch.float32, device=self.device)
        joint_angles = joint_angles.to(dtype=torch.float32, device=self.device)
        
        B = joint_angles.shape[0]
        N = points.shape[1]
        
        # Prepare input features
        sin_q = torch.sin(joint_angles)
        cos_q = torch.cos(joint_angles)
        
        # Expand joint features for each point
        joint_features = torch.cat([
            joint_angles.unsqueeze(1).expand(-1, N, -1),
            sin_q.unsqueeze(1).expand(-1, N, -1),
            cos_q.unsqueeze(1).expand(-1, N, -1)
        ], dim=-1)  # [B, N, 18]
        
        # Concatenate with points
        inputs = torch.cat([joint_features, points], dim=-1)  # [B, N, 21]
        
        if return_gradients:
            inputs.requires_grad_(True)
        
        # Forward pass
        cdf_values = self.model(inputs.reshape(B*N, -1)).reshape(B, N)
        
        if return_gradients:
            gradients = torch.zeros((B, N, 6), device=self.device)
            
            for b in range(B):
                for n in range(N):
                    try:
                        grad = torch.autograd.grad(
                            cdf_values[b, n],
                            inputs,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        
                        if grad is not None:
                            # Extract gradients for original joint angles only
                            gradients[b, n] = grad[b, n, :6]
                    except Exception as e:
                        print(f"Warning: Gradient computation failed for batch {b}, point {n}: {e}")
            
            return cdf_values, gradients
        
        return cdf_values

    def test_differentiability(self):
        """Test if the model is differentiable"""
        # Create test inputs
        points = torch.randn(1, 3, 3, device=self.device)  # [batch, num_points, 3]
        joint_angles = torch.zeros(1, 6, device=self.device, requires_grad=True)
        
        # Forward pass
        cdf = self.query_cdf(points, joint_angles)
        
        # Check if gradients flow
        try:
            loss = cdf.sum()
            loss.backward()
            print("Gradient computation successful!")
            print(f"Joint angles gradient: {joint_angles.grad}")
        except Exception as e:
            print(f"Gradient computation failed: {str(e)}")

if __name__ == "__main__":
    device = 'cuda'
    robot_cdf = RobotCDF(device)
    
    # Test points (3D)
    test_points = torch.tensor([
        [0.3, 0.2, 0.1],
        [0.4, 0.3, 0.2],
        [0.5, 0.4, 0.3]
    ], device=device).unsqueeze(0)  # [1, 3, 3]
    
    # Test joint angles
    joint_angles = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    
    print("\nTesting CDF query...")
    print(f"Test points shape: {test_points.shape}")
    print(f"Joint angles shape: {joint_angles.shape}")
    
    # Query CDF values and gradients
    cdf_values, gradients = robot_cdf.query_cdf(test_points, joint_angles, return_gradients=True)
    
    print("\nResults:")
    for i in range(len(test_points[0])):
        print(f"\nPoint {test_points[0,i].cpu().numpy()}:")
        print(f"CDF value: {cdf_values[0,i].item():.6f}")
        print(f"Gradients: {gradients[0,i].cpu().numpy()}")
    
    # Test differentiability
    robot_cdf.test_differentiability() 