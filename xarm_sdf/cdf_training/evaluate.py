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
from train_online_batch import compute_cdf_values

def evaluate_sdf_cdf_correlation(device='cuda'):
    """
    Evaluate correlation between SDF and CDF values with visualization
    """
    print("Initializing models...")
    robot_sdf = RobotSDF(device=device)
    robot_cdf = RobotCDF(device=device)
    robot_fk = XArmFK(device=device)
    visualizer = SDFVisualizer(device=device)
    
    # Load contact database
    contact_db_path = "data/cdf_data/refined_bfgs_100_contact_db.npy"
    contact_db = np.load(contact_db_path, allow_pickle=True).item()
    valid_points = torch.tensor(contact_db['points'], device=device)
    contact_configs = contact_db['contact_configs']
    link_indices = contact_db['link_indices']
    
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
        
        points = test_points[0].unsqueeze(0)
        
        # Find closest points in contact database
        closest_indices = []
        for point in points[0]:
            distances = torch.norm(valid_points - point, dim=1)
            closest_idx = torch.argmin(distances).item()
            closest_indices.append(closest_idx)
        
        # Get corresponding contact configs and link indices
        contact_configs_batch = [contact_configs[idx] for idx in closest_indices]
        link_indices_batch = [link_indices[idx] for idx in closest_indices]
        
        with torch.no_grad():
            sdf_values = robot_sdf.query_sdf(points, q.unsqueeze(0))
            cdf_values = robot_cdf.query_cdf(points, q.unsqueeze(0))
            gt_cdf_values = compute_cdf_values(
                points=points[0],
                configs=q.unsqueeze(0),
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
        
        # Show each point individually
        for j, point in enumerate(points[0]):
            print(f"\nPoint {j+1} at {point.cpu().numpy()}:")
            print(f"  SDF: {sdf_values[0,j].item():.4f}")
            print(f"  CDF (predicted): {cdf_values[0,j].item():.4f}")
            print(f"  CDF (ground truth): {gt_cdf_values[j,0].item():.4f}")
            print(f"  Contact Link Index: {link_indices_batch[j][0]}")  # Show link index for closest contact
            
            # Visualize current and closest contact config
            print("\nShowing visualizations... (close windows to continue)")
            
            # Current configuration
            scene_current = visualizer.visualize_sdf(q.unsqueeze(0), show_meshes=False, resolution=64)
            sphere_current = trimesh.primitives.Sphere(radius=0.02)
            sphere_current.visual.face_colors = [255, 0, 0, 255]  # Red color
            sphere_current.apply_translation(point.cpu().numpy())
            scene_current.add_geometry(sphere_current)
            scene_current.show()
            
            # Closest contact configuration
            contact_q = contact_configs_batch[j][0]  # Take first (closest) contact config
            contact_q_tensor = torch.tensor(contact_q, device=device).unsqueeze(0)
            scene_contact = visualizer.visualize_sdf(contact_q_tensor, show_meshes=False, resolution=64)
            sphere_contact = trimesh.primitives.Sphere(radius=0.02)
            sphere_contact.visual.face_colors = [0, 255, 0, 255]  # Green color
            sphere_contact.apply_translation(point.cpu().numpy())
            scene_contact.add_geometry(sphere_contact)
            scene_contact.show()


if __name__ == "__main__":
    evaluate_sdf_cdf_correlation() 