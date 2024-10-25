import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import List, Tuple, Optional
from data.arm_2d_config import NUM_LINKS, shapes
import random
from jax import vmap, jit

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

def create_obstacles(num_points: int = 100, rng=None) -> List[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(12345)

    obstacles = []
    
    def random_position(quadrant):
        x_range, y_range = {
            1: ((2, 3), (2, 3)),
            2: ((-7, -6), (1.5, 2.5)),
            3: ((-4, -2), (5, 6)),
            4: ((-0.5, 0.5), (3, 4))
        }[quadrant]
        return rng.uniform(*x_range), rng.uniform(*y_range)

    # Obstacle 1: Circle in first quadrant
    center = random_position(1)
    obstacles.append(create_circle(center, radius=rng.uniform(0.3, 0.6), num_points=num_points))

    # Obstacle 2: Ellipse in second quadrant
    center = random_position(2)
    obstacles.append(create_ellipse(center, a=rng.uniform(0.5, 0.8), b=rng.uniform(0.2, 0.5), 
                                    angle=rng.uniform(0, np.pi), num_points=num_points))

    # Obstacle 3: Triangle in third quadrant
    center = random_position(3)
    vertices = [
        (center[0] - random.uniform(0.5, 0.8), center[1] - random.uniform(0.5, 0.8)),
        (center[0] + random.uniform(0.5, 0.8), center[1] - random.uniform(0.5, 0.8)),
        (center[0] + random.uniform(-0.3, 0.3), center[1] + random.uniform(0.5, 0.8))
    ]
    obstacles.append(create_polygon(vertices, num_points=num_points))

    # Obstacle 4: Hexagon in fourth quadrant
    center = random_position(4)
    hexagon_vertices = []
    for i in range(6):
        angle = i * (2 * np.pi / 6)
        x = center[0] + 0.6 * np.cos(angle)
        y = center[1] + 0.6 * np.sin(angle)
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
    # joint_positions = forward_kinematics(arm_angles)
    # current_angle = 0
    
    # for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
    #     if i < len(arm_angles):
    #         current_angle += arm_angles[i]
    #     transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
    #     ax.fill(*zip(*transformed_shape), alpha=0.5)
    #     ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)


    arm_x, arm_y = forward_kinematics(arm_angles)
    # Plot the arm as a single line
    ax.plot(arm_x, arm_y, 'b-', linewidth=2, label='Robot Arm')
    
    # Plot joints
    ax.plot(arm_x[0], arm_y[0], 'ro', markersize=10, label='Base')
    ax.plot(arm_x[1:-1], arm_y[1:-1], 'ko', markersize=8, label='Joints')
    ax.plot(arm_x[-1], arm_y[-1], 'go', markersize=8, label='End effector')

    num_links = arm_angles.shape[0]
    ax.set_xlim(-num_links * 2 -2 , num_links * 2 + 2)
    ax.set_ylim(-num_links * 2 -2 , num_links * 2 + 2)
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

def visualize_simple_sdf_theta1_theta2(obstacles, resolution=200, save_path='simple_sdf_theta1_theta2_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Generate angles for theta1 and theta2
    theta1_range = jnp.linspace(-jnp.pi, jnp.pi, resolution)
    theta2_range = jnp.linspace(-jnp.pi, jnp.pi, resolution)
    theta1_mesh, theta2_mesh = jnp.meshgrid(theta1_range, theta2_range)
    
    # Flatten the meshgrid for vectorized computation
    theta1_flat = theta1_mesh.flatten()
    theta2_flat = theta2_mesh.flatten()
    
    @jit
    def forward_kinematics(theta1, theta2):
        x1 = 2 * jnp.cos(theta1)
        y1 = 2 * jnp.sin(theta1)
        x2 = x1 + 2 * jnp.cos(theta1 + theta2)
        y2 = y1 + 2 * jnp.sin(theta1 + theta2)
        return jnp.array([[0, x1, x2], [0, y1, y2]])

    @jit
    def point_to_segment_distance(p, a, b):
        ab = b - a
        ap = p - a
        proj = jnp.dot(ap, ab) / jnp.dot(ab, ab)
        proj = jnp.clip(proj, 0, 1)
        closest = a + proj * ab
        return jnp.linalg.norm(p - closest)

    @jit
    def calculate_sdf(theta1, theta2, obstacle_points):
        arm_points = forward_kinematics(theta1, theta2)
        
        d1 = vmap(lambda p: point_to_segment_distance(p, arm_points[:, 0], arm_points[:, 1]))(obstacle_points)
        d2 = vmap(lambda p: point_to_segment_distance(p, arm_points[:, 1], arm_points[:, 2]))(obstacle_points)
        
        return jnp.min(jnp.minimum(d1, d2))

    # Combine all obstacle points
    all_obstacle_points = jnp.concatenate(obstacles, axis=0)

    # Vectorize SDF calculation over all configurations
    vectorized_sdf = vmap(lambda t1, t2: calculate_sdf(t1, t2, all_obstacle_points))
    
    # Calculate SDF for all configurations
    sdf_values = vectorized_sdf(theta1_flat, theta2_flat).reshape(resolution, resolution)

    # Plot the heatmap
    heatmap = ax.imshow(sdf_values, extent=[-jnp.pi, jnp.pi, -jnp.pi, jnp.pi], origin='lower', 
                        aspect='auto', cmap='viridis')
    
    # Highlight the 0.05 level set
    level_set = ax.contour(theta1_mesh, theta2_mesh, sdf_values, levels=[0.05], colors='red', linewidths=2)
    ax.clabel(level_set, inline=True, fmt='0.05', colors='red', fontsize=10)
    
    ax.set_xlabel('Theta 1 (radians)')
    ax.set_ylabel('Theta 2 (radians)')
    ax.set_title('Simple SDF Visualization for Theta 1 and Theta 2')
    
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Minimum Distance to Obstacles')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")

@jit
def forward_kinematics(joint_angles):
    num_links = len(joint_angles)
    link_length = 2.0  # Assuming each link has a length of 2 units
    
    # Initialize arrays to store x and y coordinates
    x = jnp.zeros(num_links + 1)
    y = jnp.zeros(num_links + 1)
    
    # Calculate cumulative angle
    cumulative_angle = jnp.cumsum(joint_angles)
    
    # Calculate x and y positions for each joint
    for i in range(1, num_links + 1):
        x = x.at[i].set(x[i-1] + link_length * jnp.cos(jnp.sum(joint_angles[:i])))
        y = y.at[i].set(y[i-1] + link_length * jnp.sin(jnp.sum(joint_angles[:i])))
    
    return x, y

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
    # visualize_sdf_theta1_theta2(params_list, obstacles)

    visualize_simple_sdf_theta1_theta2(obstacles)


if __name__ == "__main__":
    main()
