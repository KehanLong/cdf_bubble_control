import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional
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

def create_obstacles(num_points: int = 50, rng=None) -> List[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(12345)

    obstacles = []
    
    def random_position(quadrant):
        x_range, y_range = {
            1: ((2, 3), (2, 3)),
            2: ((2, 3), (-2.5, -1.5)),
            3: ((-3, -2), (-2.5, -2)),
            4: ((-3.5, -3.1), (3.1, 3.5))
        }[quadrant]
        return rng.uniform(*x_range), rng.uniform(*y_range)

    # Obstacle 1: Circle in first quadrant
    center = random_position(1)
    obstacles.append(create_circle(center, radius=rng.uniform(0.6, 0.8), num_points=num_points))

    # Obstacle 2: Ellipse in 4th quadrant
    center = random_position(2)
    obstacles.append(create_ellipse(center, a=rng.uniform(0.7, 0.8), b=rng.uniform(0.4, 0.6), 
                                    angle=rng.uniform(0, np.pi), num_points=num_points))
    #obstacles.append(create_circle(center, radius=rng.uniform(0.3, 0.6), num_points=num_points))

    # Obstacle 3: Triangle in third quadrant
    center = random_position(3)
    vertices = [
        (center[0] - random.uniform(0.7, 0.9), center[1] - random.uniform(0.5, 0.8)),
        (center[0] + random.uniform(0.7, 0.9), center[1] - random.uniform(0.5, 0.8)),
        (center[0] + random.uniform(-0.3, 0.3), center[1] + random.uniform(0.5, 0.8))
    ]
    obstacles.append(create_polygon(vertices, num_points=num_points))
    #obstacles.append(create_circle(center, radius=rng.uniform(0.3, 0.6), num_points=num_points))

    # Obstacle 4: Hexagon in 2nd quadrant
    center = random_position(4)
    hexagon_vertices = []
    for i in range(6):
        angle = i * (2 * np.pi / 6)
        x = center[0] + 0.6 * np.cos(angle)
        y = center[1] + 0.6 * np.sin(angle)
        hexagon_vertices.append((x, y))
    obstacles.append(create_polygon(hexagon_vertices, num_points=num_points))
    

    return obstacles

def plot_environment(obstacles: List[np.ndarray], arm_angles: np.ndarray, 
                    ax: Optional[plt.Axes] = None, save_path: Optional[str] = None,
                    goal_angles: Optional[np.ndarray] = None,
                    robot_color: str = 'blue', robot_alpha: float = 1.0,
                    plot_obstacles: bool = True, add_to_legend: bool = True,
                    label: Optional[str] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

    # Plot obstacles
    if plot_obstacles:
        for obstacle in obstacles:
            ax.fill(obstacle[:, 0], obstacle[:, 1], alpha=0.5)
            ax.scatter(obstacle[:, 0], obstacle[:, 1], color='red', s=1, alpha=0.8)

    # Plot current configuration
    arm_x, arm_y = forward_kinematics(arm_angles)
    ax.plot(arm_x, arm_y, color=robot_color, linestyle='-', 
            linewidth=5, alpha=robot_alpha, 
            label=label if add_to_legend and label else None)
    
    # Only plot markers if this is not an intermediate configuration
    if robot_alpha > 0.9:  # Only for initial/goal configs
        ax.plot(arm_x[0], arm_y[0], 'bo', markersize=15)
        ax.plot(arm_x[1:-1], arm_y[1:-1], 'ko', markersize=15)
        ax.plot(arm_x[-1], arm_y[-1], 'go', markersize=15)

    # Plot goal configuration if provided
    if goal_angles is not None:
        goal_x, goal_y = forward_kinematics(goal_angles)
        ax.plot(goal_x, goal_y, 'r--', linewidth=5, label='Goal Config')
        ax.plot(goal_x[1:-1], goal_y[1:-1], 'ko', markersize=15)
        ax.plot(goal_x[-1], goal_y[-1], 'ro', markersize=15)

    num_links = arm_angles.shape[0]
    ax.set_xlim(-num_links * 2 -1 , num_links * 2 + 1)
    ax.set_ylim(-num_links * 2 -1 , num_links * 2 + 1)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)

    plt.tight_layout()
    
    # Only add legend for non-intermediate configurations
    if add_to_legend:
        ax.legend(fontsize=26)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    
    return fig, ax

def forward_kinematics(joint_angles):
    # Convert input to torch tensor if it's not already
    if not isinstance(joint_angles, torch.Tensor):
        joint_angles = torch.tensor(joint_angles, dtype=torch.float32)
    
    num_links = len(joint_angles)
    link_length = 2.0  # Assuming each link has a length of 2 units
    
    # Initialize arrays to store x and y coordinates
    x = torch.zeros(num_links + 1)
    y = torch.zeros(num_links + 1)
    
    # Calculate x and y positions for each joint
    for i in range(1, num_links + 1):
        x[i] = x[i-1] + link_length * torch.cos(torch.sum(joint_angles[:i]))
        y[i] = y[i-1] + link_length * torch.sin(torch.sum(joint_angles[:i]))
    
    return x.numpy(), y.numpy()

def main():

    # Create sample obstacles with point cloud observations
    num_points = 50  # You can adjust this value
    obstacles = create_obstacles(num_points)

    # Create a sample arm configuration (5 joints)
    arm_angles = np.array([0, 0])

    # Plot the environment
    fig, ax = plot_environment(obstacles, arm_angles)
    plt.show()

    print("Sample obstacles created:")
    for i, obstacle in enumerate(obstacles):
        print(f"Obstacle {i+1}: {len(obstacle)} points")

    print(f"\nSample arm configuration: {arm_angles}")
    


if __name__ == "__main__":
    main()
