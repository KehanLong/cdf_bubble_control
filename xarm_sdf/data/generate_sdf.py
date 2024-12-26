import trimesh
import glob
import os
import numpy as np
import mesh_to_sdf
import torch
from pathlib import Path
import matplotlib.pyplot as plt

import sys
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from models.xarm6_differentiable_fk import fk_xarm6_torch


def generate_sdf_data(mesh_dir, output_dir, num_points=100000):
    """Generate SDF data for each link of the xArm robot"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    mesh_files = glob.glob(os.path.join(mesh_dir, "*.stl"))
    mesh_files = sorted(mesh_files)
    
    print(f"Found {len(mesh_files)} mesh files in {mesh_dir}")
    print("Mesh files:", [Path(f).name for f in mesh_files])
    
    # Create list to store output files
    output_files = []
    
    for link_id, mf in enumerate(mesh_files):
        mesh_name = Path(mf).stem
        print(f"\nProcessing {mesh_name}")
        
        # Load mesh and store original scale
        mesh = trimesh.load(mf)
        original_center = mesh.bounding_box.centroid
        original_scale = np.max(np.linalg.norm(mesh.vertices-original_center, axis=1))
        
        # Now normalize for SDF computation
        mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
        
        # Get normalized mesh metadata
        center = mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(mesh.vertices-center, axis=1))
        
        print(f"Original scale: {original_scale}, Normalized scale: {scale}")
        
        # Sample points
        near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            number_of_points=num_points,
            surface_point_method='scan',
            sign_method='normal',
            scan_count=100,
            scan_resolution=400,
            sample_point_count=2000000,
            normal_sample_count=100,
            min_size=0.01,
            return_gradients=False
        )
        
        random_points = np.random.rand(num_points, 3) * 2.0 - 1.0
        random_sdf = mesh_to_sdf.mesh_to_sdf(
            mesh,
            random_points,
            surface_point_method='scan',
            sign_method='normal',
            scan_count=100,
            scan_resolution=400,
            sample_point_count=2000000,
            normal_sample_count=100
        )
        
        data = {
            'near_points': near_points,
            'near_sdf': near_sdf,
            'random_points': random_points,
            'random_sdf': random_sdf,
            'center': center,
            'scale': scale,
            'original_center': original_center,
            'original_scale': original_scale,
            'link_id': link_id
        }
        
        output_file = os.path.join(output_dir, f'link_{link_id}_{mesh_name}.npz')
        np.savez_compressed(output_file, **data)
        print(f"Saved {output_file}")
        
        output_files.append(output_file)

    return output_files



def visualize_robot_sdf(data_dir, num_links=7, joint_angles=None):
    """
    Visualize robot links from saved SDF data
    Args:
        data_dir: directory containing the .npz files
        num_links: number of links to visualize (1-7)
        joint_angles: robot configuration (optional, defaults to zero pose)
    """
    if joint_angles is None:
        joint_angles = torch.zeros(6)
    
    # Find all .npz files
    data_files = sorted(glob.glob(str(data_dir / "*.npz")))
    if not data_files:
        print("No data files found!")
        return
        
    # Limit data files to requested number of links
    data_files = data_files[:num_links]
    
    print(f"\nVisualizing {num_links} links (including base)")
    print(f"Joint angles: {joint_angles.numpy()}")
    
    # Get transforms using our FK function
    transforms = fk_xarm6_torch(joint_angles)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process each link
    colors = ['gray', 'blue', 'red', 'green', 'cyan', 'magenta', 'yellow']
    for i, file_path in enumerate(data_files):
        data = np.load(file_path)
        near_points = data['near_points']
        original_center = data['original_center']
        original_scale = data['original_scale']
        
        print(f"\nProcessing Link {i}")
        print(f"Original center: {original_center}")
        print(f"Original scale: {original_scale}")
        
        # Scale back to original size
        near_points = near_points * original_scale + original_center
        
        if i == 0:  # Base link
            transformed_points = near_points
        else:
            # Convert to homogeneous coordinates
            points_h = np.ones((near_points.shape[0], 4))
            points_h[:, :3] = near_points
            points_h = torch.FloatTensor(points_h)
            
            # Transform points using our FK
            transform = transforms[i-1].numpy()
            transformed_points = points_h @ transform.T
            transformed_points = transformed_points[:, :3]
        
        ax.scatter(transformed_points[:, 0], 
                  transformed_points[:, 1], 
                  transformed_points[:, 2],
                  c=colors[i], s=1, alpha=0.3, label=f'Link {i}')
        
        if i > 0:
            # Draw coordinate frames at each joint
            origin = transforms[i-1][:3, 3].numpy()
            for axis, c in zip(range(3), ['r', 'g', 'b']):
                direction = transforms[i-1][:3, axis].numpy()
                ax.quiver(origin[0], origin[1], origin[2],
                         direction[0], direction[1], direction[2],
                         length=0.1, color=c, alpha=0.5)
    
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Set reasonable axis limits based on robot dimensions
    limit = 0.5  # 50cm in each direction
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([0, 2*limit])
    
    plt.title(f'xArm Robot SDF - {num_links} Links')
    plt.show()


'''
the following code is for debugging the sdf data generation
'''

def visualize_sdf_data(data):
    """
    Visualize SDF data for a single link
    Args:
        data: dictionary containing SDF data
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot near-surface points
    ax1 = fig.add_subplot(131, projection='3d')
    near_points = data['near_points']
    near_sdf = data['near_sdf']
    
    # Color points by SDF value
    scatter = ax1.scatter(near_points[:, 0], near_points[:, 1], near_points[:, 2], 
                         c=near_sdf, cmap='RdBu', s=1)
    plt.colorbar(scatter, ax=ax1)
    ax1.set_title('Near Surface Points')
    
    # Plot random points
    ax2 = fig.add_subplot(132, projection='3d')
    random_points = data['random_points']
    random_sdf = data['random_sdf']
    
    scatter = ax2.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], 
                         c=random_sdf, cmap='RdBu', s=1)
    plt.colorbar(scatter, ax=ax2)
    ax2.set_title('Random Points')
    
    # Plot histogram of SDF values
    ax3 = fig.add_subplot(133)
    ax3.hist(near_sdf, bins=50, alpha=0.5, label='Near Surface')
    ax3.hist(random_sdf, bins=50, alpha=0.5, label='Random')
    ax3.set_title('SDF Value Distribution')
    ax3.legend()
    
    # Print some statistics
    print(f"Number of near surface points: {len(near_points)}")
    print(f"Number of random points: {len(random_points)}")
    print(f"Near surface SDF range: [{near_sdf.min():.3f}, {near_sdf.max():.3f}]")
    print(f"Random SDF range: [{random_sdf.min():.3f}, {random_sdf.max():.3f}]")
    print(f"Center: {data['center']}")
    print(f"Scale: {data['scale']}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    mesh_dir = project_root / "xarm_description" / "meshes" / "xarm6" / "visual"
    output_dir = project_root / "data" / "sdf_data"
    
    # Comment out data generation when not needed
    # generate_sdf_data(str(mesh_dir), str(output_dir))
    
    # Visualize with different configurations
    zero_angles = torch.zeros(6)
    bent_angles = torch.tensor([np.pi/4, np.pi/4, -np.pi/4, 0.0, 0.0, 0.0])
    
    visualize_robot_sdf(output_dir, num_links=7, joint_angles=zero_angles)
    visualize_robot_sdf(output_dir, num_links=7, joint_angles=bent_angles)
