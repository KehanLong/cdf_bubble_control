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
    Plot 3D points together with the robot visualization
    Args:
        points: Nx3 array of points
        joint_angles: tensor of 6 joint angles in radians
        title: plot title
        color: point cloud color
        resolution: resolution for robot SDF visualization
    """
    # Initialize visualizer
    visualizer = SDFVisualizer('cuda')
    
    # Create scene with robot
    scene = visualizer.create_scene(joint_angles, show_meshes=True, resolution=resolution)
    
    # Add point cloud to the scene
    point_cloud = trimesh.PointCloud(points, colors=[0, 0, 255, 255])  # Blue points
    scene.add_geometry(point_cloud)
    
    # Show scene
    scene.show()

def main():
    # Get the directory containing point cloud data
    data_dir = os.path.join(os.path.dirname(__file__))
    
    # Specifically look for final_downsampled_points.npy
    filename = "final_downsampled_points.npy"
    filepath = os.path.join(data_dir, filename)
    
    try:
        points = np.load(filepath)
        
        # If points include RGB values (4th column), remove it
        if points.shape[1] > 3:
            points = points[:, :3]
        
        # Print original points info
        print("Original points:")
        print_points_info(points, filename)
        
        # Filter points with y >= 0.2
        filtered_points = points[points[:, 1] >= 0.2]
        
        # Print filtered points info
        print("\nFiltered points (y >= 0.2):")
        print_points_info(filtered_points, filename)
        
        # Save filtered points
        filtered_filepath = os.path.join(data_dir, "filtered_final_points.npy")
        np.save(filtered_filepath, filtered_points)
        print(f"\nSaved filtered points to: {filtered_filepath}")
        
        # Visualize both original and filtered points
        plot_points(points, title="Original Points")
        plot_points(filtered_points, title="Filtered Points (y >= 0.2)")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()