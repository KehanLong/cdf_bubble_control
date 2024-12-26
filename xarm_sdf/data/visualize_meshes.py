import trimesh
import glob
from pathlib import Path
import numpy as np
import torch


import sys
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from models.xarm6_differentiable_fk import fk_xarm6_torch

def visualize_robot_meshes(mesh_dir, q):
    """
    Visualize robot mesh at a specific joint configuration
    Args:
        mesh_dir: directory containing .stl files
        q: joint angles in radians [q1, q2, q3, q4, q5, q6]
    """
    # Load meshes
    mesh_files = sorted(glob.glob(str(mesh_dir / "*.stl")))
    meshes = [trimesh.load(mf) for mf in mesh_files]
    
    # Create a scene
    scene = trimesh.Scene()
    
    # Assign different colors to different links
    colors = [[128,128,128,150], [255,0,0,150], [0,255,0,150], 
             [0,0,255,150], [255,255,0,150], [255,0,255,150], 
             [0,255,255,150]]
    
    # Get transforms using our FK function
    transforms = fk_xarm6_torch(q)
    transforms_dict = {
        'base': np.eye(4),
        'link1': transforms[0].numpy(),
        'link2': transforms[1].numpy(),
        'link3': transforms[2].numpy(),
        'link4': transforms[3].numpy(),
        'link5': transforms[4].numpy(),
        'link6': transforms[5].numpy()
    }
    
    # Add meshes with transforms
    for i, mesh in enumerate(meshes):
        link_name = Path(mesh_files[i]).stem.split('_')[0]
        if link_name in transforms_dict:
            mesh.apply_transform(transforms_dict[link_name])
        mesh.visual.face_colors = colors[i % len(colors)]
        scene.add_geometry(mesh)
    
    # Show the scene
    scene.show()

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    mesh_dir = project_root / "xarm_description" / "meshes" / "xarm6" / "visual"
    
    # Try different poses
    poses = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Home pose
        torch.tensor([np.pi/4, 0, -np.pi/2, 0.0, 0.0, 0.0]),  # Bent pose
        torch.tensor([np.pi/2, 0.0, -np.pi/4, np.pi/4, np.pi/2, 0.0])  # Another pose
    ]
    
    # Visualize each pose
    for i, pose in enumerate(poses):
        print(f"\nVisualizing pose {i+1}:")
        print(f"Joint angles (radians): {pose.numpy()}")
        visualize_robot_meshes(mesh_dir, pose) 