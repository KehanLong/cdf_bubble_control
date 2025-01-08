import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path

def visualize_contact_points(contact_db_path):
    """Visualize valid points from contact database"""
    # Load the contact database
    contact_db = np.load(contact_db_path, allow_pickle=True).item()
    points = contact_db['points']  # [N, 3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of points
    scatter = ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2],
        c=points[:, 2],  # Color by height
        cmap='viridis',
        alpha=0.6,
        s=1  # Point size
    )
    
    # Add colorbar
    plt.colorbar(scatter, label='Z Height')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title with point count
    ax.set_title(f'Valid Contact Points (n={len(points)})')
    
    # Add grid
    ax.grid(True)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save plot
    save_dir = Path(contact_db_path).parent / 'visualizations'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'valid_contact_points.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_dir / 'valid_contact_points.png'}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\nPoint Statistics:")
    print(f"Total points: {len(points)}")
    print("\nBounding Box:")
    print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

if __name__ == "__main__":
    # Assuming standard project structure
    project_root = Path(__file__).parent.parent
    contact_db_path = project_root / "data/cdf_data/contact_db.npy"
    
    visualize_contact_points(contact_db_path)