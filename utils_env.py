import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import List, Tuple, Optional
from arm_2d_utils import forward_kinematics, transform_shape, generate_robot_point_cloud, calculate_arm_sdf
from data.arm_2d_config import NUM_LINKS, shapes
import random

def create_circle(center: Tuple[float, float], radius: float, num_points: int = 100) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack((x, y))

def create_polygon(vertices: List[Tuple[float, float]], num_points: int = 100) -> np.ndarray:
    points = []
    num_edges = len(vertices)
    points_per_edge = num_points // num_edges
    remaining_points = num_points % num_edges

    for i in range(num_edges):
        start = np.array(vertices[i])
        end = np.array(vertices[(i+1) % num_edges])
        edge_points = points_per_edge + (1 if i < remaining_points else 0)
        t = np.linspace(0, 1, edge_points)
        segment_points = start[None, :] + t[:, None] * (end - start)[None, :]
        points.append(segment_points)
    
    return np.vstack(points)

def create_ellipse(center: Tuple[float, float], a: float, b: float, angle: float = 0, num_points: int = 100) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_points = np.dot(rotation_matrix, np.vstack((x, y)))
    return np.column_stack((rotated_points[0] + center[0], rotated_points[1] + center[1]))

def create_obstacles(num_points: int = 100) -> List[np.ndarray]:
    obstacles = []
    
    # Helper function to get a random position in a quadrant
    def random_position(quadrant):
        x_range, y_range = {
            1: ((5, 8), (5, 8)),
            2: ((-8, -5), (5, 8)),
            3: ((-12, -10), (-12, -10)),
            4: ((5, 8), (-8, -5))
        }[quadrant]
        return random.uniform(*x_range), random.uniform(*y_range)

    # Obstacle 1: Circle in first quadrant
    center = random_position(1)
    obstacles.append(create_circle(center, radius=random.uniform(2.5, 3.5), num_points=num_points))

    # Obstacle 2: Ellipse in second quadrant
    center = random_position(2)
    obstacles.append(create_circle(center, radius=random.uniform(2.5, 3.5), num_points=num_points))

    # Obstacle 3: Triangle in third quadrant
    # center = random_position(3)
    # vertices = [
    #     (center[0] - random.uniform(2, 3), center[1] - random.uniform(2, 3)),
    #     (center[0] + random.uniform(2, 3), center[1] - random.uniform(2, 3)),
    #     (center[0] + random.uniform(-1, 1), center[1] + random.uniform(2, 3))
    # ]
    # obstacles.append(create_polygon(vertices, num_points=num_points))

    # Obstacle 4: Hexagon in fourth quadrant
    center = random_position(4)
    hexagon_vertices = []
    for i in range(6):
        angle = i * (2 * np.pi / 6)
        x = center[0] + 2 * np.cos(angle)
        y = center[1] + 2 * np.sin(angle)
        hexagon_vertices.append((x, y))
    obstacles.append(create_polygon(hexagon_vertices, num_points=num_points))

    return obstacles

def plot_environment(obstacles: List[np.ndarray], arm_angles: np.ndarray, ax: Optional[plt.Axes] = None, save_path: Optional[str] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

    # Plot obstacles
    for obstacle in obstacles:
        ax.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.5)
        # Plot point cloud observations on obstacle surfaces
        ax.scatter(obstacle[:, 0], obstacle[:, 1], color='red', s=1, alpha=0.8)

    # Plot robot arm
    joint_positions = forward_kinematics(arm_angles)
    current_angle = 0
    
    for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
        if i < len(arm_angles):
            current_angle += arm_angles[i]
        transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
        ax.fill(*zip(*transformed_shape), alpha=0.5)
        ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Robot Arm, Obstacles, and Point Cloud Observations')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")

    # if show_plot:
    #     plt.show()
    
    return fig, ax

def visualize_sdf_theta1_theta2(params_list, obstacles, resolution=200, save_path='sdf_theta1_theta2_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Generate angles for theta1 and theta2
    theta1_range = np.linspace(-np.pi, np.pi, resolution)
    theta2_range = np.linspace(-np.pi/2, np.pi/2, resolution)
    theta1_mesh, theta2_mesh = np.meshgrid(theta1_range, theta2_range)
    
    # Flatten the meshgrid for vectorized computation
    theta1_flat = theta1_mesh.flatten()
    theta2_flat = theta2_mesh.flatten()
    
    # Create angles array with only two angles
    angles_flat = jnp.column_stack((theta1_flat, theta2_flat))
    
    # Combine all obstacle points
    all_obstacle_points = jnp.concatenate(obstacles, axis=0)
    
    # Compute SDF for all configurations
    sdf_values = []
    for angles in angles_flat:
        # Calculate SDF for all obstacle points
        distances, _ = calculate_arm_sdf(all_obstacle_points, angles, params_list)
        # Take the minimum distance as the SDF value for this configuration
        sdf_values.append(jnp.min(distances))
    
    sdf_values = jnp.array(sdf_values).reshape(resolution, resolution)
    
    # Plot the heatmap
    heatmap = ax.imshow(sdf_values, extent=[-np.pi, np.pi, -np.pi/2, np.pi/2], origin='lower', 
                        aspect='auto', cmap='viridis')
    contour = ax.contour(theta1_mesh, theta2_mesh, sdf_values, levels=[0], colors='red', linewidths=2)
    
    ax.set_xlabel('Theta 1 (radians)')
    ax.set_ylabel('Theta 2 (radians)')
    ax.set_title('SDF Visualization for Theta 1 and Theta 2')
    
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Minimum SDF Value')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    # plt.show()


def main():

    # Create sample obstacles with point cloud observations
    num_points = 50  # You can adjust this value
    obstacles = create_obstacles(num_points)

    # Create a sample arm configuration (5 joints)
    arm_angles = np.array([0, 0, 0, 0, 0])

    # Plot the environment
    plot_environment(obstacles, arm_angles)

    print("Sample obstacles created:")
    for i, obstacle in enumerate(obstacles):
        print(f"Obstacle {i+1}: {len(obstacle)} points")

    print(f"\nSample arm configuration: {arm_angles}")

    # Load params_list (assuming you have this part in your main function)
    params_list = []
    for i in range(NUM_LINKS):
        params = jnp.load(f"trained_models/sdf_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)
    
    # Visualize SDF for theta1 and theta2
    visualize_sdf_theta1_theta2(params_list, obstacles)


if __name__ == "__main__":
    main()