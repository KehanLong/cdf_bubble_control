import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from arm_2d_utils import forward_kinematics, transform_shape
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
            1: ((4, 8), (4, 8)),
            2: ((-15, -5), (5, 15)),
            3: ((-15, -5), (-15, -5)),
            4: ((5, 15), (-15, -5))
        }[quadrant]
        return random.uniform(*x_range), random.uniform(*y_range)

    # Obstacle 1: Circle in first quadrant
    center = random_position(1)
    obstacles.append(create_circle(center, radius=random.uniform(2.5, 3.5), num_points=num_points))

    # Obstacle 2: Ellipse in second quadrant
    center = random_position(2)
    obstacles.append(create_ellipse(center, a=random.uniform(3, 4), b=random.uniform(1.5, 2.5), 
                                    angle=random.uniform(0, np.pi), num_points=num_points))

    # Obstacle 3: Triangle in third quadrant
    center = random_position(3)
    vertices = [
        (center[0] - random.uniform(2, 3), center[1] - random.uniform(2, 3)),
        (center[0] + random.uniform(2, 3), center[1] - random.uniform(2, 3)),
        (center[0] + random.uniform(-1, 1), center[1] + random.uniform(2, 3))
    ]
    obstacles.append(create_polygon(vertices, num_points=num_points))

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

def plot_environment(obstacles: List[np.ndarray], arm_angles: np.ndarray, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
    else:
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

    if show_plot:
        plt.show()

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

if __name__ == "__main__":
    main()