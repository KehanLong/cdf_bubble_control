import os
import numpy as np
import trimesh
import sys
from pathlib import Path
import torch

import matplotlib.pyplot as plt

# Add project root to path to import visualization
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.visualization import SDFVisualizer

def print_points_info(points, filename):
    """Print statistical information about the point cloud"""
    print(f"\nFile: {filename}")
    print(f"Number of points: {len(points)}")
    print(f"Point cloud bounds:")
    print(f"X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
    print(f"Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
    print(f"Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

def plot_points(points, title="Point Cloud", color='b'):
    """
    Plot 3D points with coordinate axes using matplotlib
    Args:
        points: Nx3 array of points
        title: plot title
        color: point cloud color
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.', s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Add a grid
    ax.grid(True)
    
    # Show plot
    plt.show()

def plot_points_with_robot(points, joint_angles, title="Point Cloud with Robot", color='b', resolution=64):
    """
    Plot 3D points together with the robot visualization using trained SDF
    Args:
        points: Nx3 array of points
        joint_angles: tensor of 6 joint angles in radians
        title: plot title
        color: point cloud color
        resolution: resolution for robot SDF visualization
    """
    # Initialize visualizer
    visualizer = SDFVisualizer('cuda')
    
    # Create scene with robot SDF
    scene = visualizer.visualize_sdf(joint_angles, show_meshes=False, resolution=resolution)
    
    # Add point cloud to the scene
    point_cloud = trimesh.PointCloud(points, colors=[0, 0, 255, 255])  # Blue points
    scene.add_geometry(point_cloud)
    
    # Show scene
    scene.show()

def main():
    # Get the directory containing point cloud data
    data_dir = os.path.dirname(__file__)
    
    # List of files to visualize
    files_to_visualize = [
        "final_downsampled_points.npy",
        "robot_filtered_points.npy",
        "transformed_points.npy"
    ]
    
    # Robot pose for visualization
    joint_angles = torch.tensor([0.0, -0.5, -0.5, 0., 0.85, 0.0], device='cuda')
    
    # Process each file
    for file_name in files_to_visualize:
        filepath = os.path.join(data_dir, file_name)
        
        try:
            if not os.path.exists(filepath):
                print(f"Could not find file: {filepath}")
                continue
                
            points = np.load(filepath)
            print(f"\nProcessing: {file_name}")
            
            # If points include RGB values (4th column), remove it
            if points.shape[1] > 3:
                points = points[:, :3]
            
            # Print information about the points
            print_points_info(points, file_name)
            
            # Visualize points with robot SDF
            plot_points_with_robot(points, 
                                 joint_angles, 
                                 title=f"Points from: {file_name}")
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")

if __name__ == "__main__":
    main()