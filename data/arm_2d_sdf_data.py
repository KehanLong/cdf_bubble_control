import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import matplotlib.pyplot as plt
from data.arm_2d_config import shapes
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

from data.dataset import SDFDataset

def point_to_segment_distance(point, segment_start, segment_end):
    """Compute the distance from a point to a line segment."""
    segment = segment_end - segment_start
    point_vec = point - segment_start
    proj = np.dot(point_vec, segment) / np.dot(segment, segment)
    proj = max(0, min(1, proj))
    closest = segment_start + proj * segment
    return np.linalg.norm(point - closest)

def is_point_inside_polygon(point, polygon):
    """Check if a point is inside a polygon using ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def compute_sdf(point, polygon):
    """Compute the signed distance from a point to a polygon."""
    if is_point_inside_polygon(point, polygon):
        return -min(point_to_segment_distance(point, polygon[i], polygon[(i+1) % len(polygon)]) 
                    for i in range(len(polygon)))
    else:
        return min(point_to_segment_distance(point, polygon[i], polygon[(i+1) % len(polygon)]) 
                   for i in range(len(polygon)))

def generate_sdf_data(shape_points, num_samples):
    """Generate SDF data for a given shape defined by its corner points."""
    # Determine bounding box
    min_x, min_y = np.min(shape_points, axis=0)
    max_x, max_y = np.max(shape_points, axis=0)
    
    # Generate sample points
    x_coords = np.random.uniform(min_x - 1, max_x + 1, num_samples)
    y_coords = np.random.uniform(min_y - 1, max_y + 1, num_samples)
    points = np.column_stack((x_coords, y_coords, np.zeros(num_samples)))

    # Compute distances
    distances = np.array([compute_sdf(point[:2], shape_points) for point in points]).reshape(-1, 1)

    return points, distances

def visualize_training_data(dataset, shape_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create custom colormap
    colors = ['darkblue', 'blue', 'lightblue', 'white', 'pink', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Normalize distances for colormap
    distances = dataset.distances.reshape(-1)
    max_abs_dist = np.max(np.abs(distances))
    norm = plt.Normalize(-max_abs_dist, max_abs_dist)
    
    # Create a grid for interpolation
    grid_x, grid_y = np.mgrid[-5:5:100j, -5:5:100j]
    
    # Interpolate the distances onto the grid
    grid_z = griddata(dataset.points[:, :2], distances, (grid_x, grid_y), method='cubic')
    
    # Plot the interpolated heatmap
    im = ax.imshow(grid_z.T, extent=[-5, 5, -5, 5], origin='lower', 
                   cmap=cmap, norm=norm, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Signed Distance')
    
    # Plot the zero level set
    ax.contour(grid_x, grid_y, grid_z, levels=[0], colors='black', linewidths=2)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f'Training Data - {shape_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

def main():
    num_samples_per_link = 7000


    for i, (shape_name, shape_points) in enumerate(shapes):
        points, distances = generate_sdf_data(np.array(shape_points), num_samples_per_link)

        dataset = SDFDataset(points, distances)
        visualize_training_data(dataset, shape_name)

        # Save the dataset for each link separately
        data = {
            'points': points,
            'distances': distances
        }
        np.save(f'link{i+1}_sdf_data.npy', data)

if __name__ == "__main__":
    main()