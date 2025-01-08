import torch
import numpy as np
from pathlib import Path
import trimesh
import matplotlib.pyplot as plt
import time

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from models.xarm_model import XArmFK
from utils.visualization import SDFVisualizer

def evaluate_sdf_cdf_correlation(device='cuda'):
    """
    Evaluate correlation between SDF and CDF values with visualization
    """
    print("Initializing models...")
    robot_sdf = RobotSDF(device=device)
    robot_cdf = RobotCDF(device=device)
    robot_fk = XArmFK(device=device)
    visualizer = SDFVisualizer(device=device)
    
    # Define test configurations
    test_configs = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device),  # Home pose
        torch.tensor([np.pi/4, np.pi/3, -np.pi/4, np.pi/6, np.pi/4, 0.0], device=device),  # Bent pose
    ]
    
    # Define test points for each configuration
    test_points = [
        # Points around the robot
        torch.tensor([
            [0.3, 0.0, 0.3],   # Front of robot
            [0.0, 0.3, 0.3],   # Side of robot
            [0.3, 0.3, 0.3],   # Diagonal from robot
            [0.2, 0.0, 0.5],   # Above robot
            [0.4, 0.0, 0.2],   # Near base
        ], device=device)
    ]
    
    # Evaluate each configuration
    for i, q in enumerate(test_configs):
        print(f"\n=== Configuration {i+1} ===")
        print(f"Joint angles: {q.cpu().numpy()}")
        
        points = test_points[0].unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            sdf_values = robot_sdf.query_sdf(points, q.unsqueeze(0))
            cdf_values = robot_cdf.query_cdf(points, q.unsqueeze(0))
        
        # Show each point individually
        for j, point in enumerate(points[0]):
            # Create new scene for each point
            scene = visualizer.visualize_sdf(q.unsqueeze(0), show_meshes=False, resolution=64)
            
            # Create sphere at point location
            sphere = trimesh.primitives.Sphere(radius=0.02)
            sphere.visual.face_colors = [255, 0, 0, 255]  # Red color
            sphere.apply_translation(point.cpu().numpy())
            scene.add_geometry(sphere)
            
            print(f"\nPoint {j+1} at {point.cpu().numpy()}:")
            print(f"  SDF: {sdf_values[0,j].item():.4f}")
            print(f"  CDF: {cdf_values[0,j].item():.4f}")
            
            # Show scene
            print("\nShowing visualization... (close window to continue)")
            scene.show()
            


if __name__ == "__main__":
    evaluate_sdf_cdf_correlation() 