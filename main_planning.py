import numpy as np
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import torch
import matplotlib.pyplot as plt
import imageio
from sdf_marching.samplers import get_rapidly_exploring
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.samplers.tracing import trace_toward_graph_all
from sdf_marching.discrete import get_shortest_path
from sdf_marching.cvx import edgeseq_to_traj_constraint_bezier, bezier_cost_all

import cvxpy

from matplotlib.patches import Circle, Rectangle

from utils_env import visualize_sdf_theta1_theta2, visualize_simple_sdf_theta1_theta2
from data.arm_2d_config import NUM_LINKS

class CDF:
    def __init__(self, weights_file, obstacles, device):
        self.model = load_learned_cdf(weights_file)
        self.model.to(device)
        self.obstacle_points = concatenate_obstacle_list(obstacles)
        self.device = device
    
    def __call__(self, configurations):
        cdf_values = cdf_evaluate_model(self.model, configurations, self.obstacle_points, self.device)
        return np.min(cdf_values, axis=1)


def animate_path(obstacles: List[np.ndarray], planned_path: np.ndarray, fps: int = 10, duration: float = 5.0):
    """
    Create an animation of the robot arm moving along the planned path.
    
    Args:
    obstacles (List[np.ndarray]): List of obstacle arrays.
    planned_path (np.ndarray): Array of shape (n_steps, 5) containing the planned configurations.
    fps (int): Frames per second for the animation.
    duration (float): Duration of the animation in seconds.
    
    Returns:
    None: Saves the animation as a video file.
    """
    n_frames = int(fps * duration)
    path_indices = np.linspace(0, len(planned_path) - 1, n_frames, dtype=int)
    
    frames = []
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in path_indices:
        ax.clear()
        config = planned_path[i]

        plot_environment(obstacles, config, ax=ax)
        ax.set_title(f"Step {i+1}/{len(planned_path)}")
        
        # Convert plot to image
        fig.canvas.draw()
        plt.show(block=False)
        # plt.pause(0.1)
        # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # frames.append(image)
        fig.savefig(f"figures/test{i}.png", bbox_inches="tight")
    
    plt.close(fig)
    
    # Save as video using imageio
    # imageio.mimsave('robot_arm_animation.mp4', frames, fps=fps)

def wrap_angles(angles: np.ndarray) -> np.ndarray:
    return np.mod(
        angles + np.pi,
        2 * np.pi
    ) - np.pi


def config_to_cdf_input(angles: np.ndarray) -> np.ndarray:
    """
    Convert arm configuration angles to the input format required by cdf_evaluate_model.
    
    Args:
    angles (np.ndarray): Array of joint angles.
    
    Returns:
    np.ndarray: Formatted input for cdf_evaluate_model.
    """
    if angles.ndim == 1:
        angles = angles[np.newaxis, :]
    
    return angles  # The encoding is now done inside cdf_evaluate_model

def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
    obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
    np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)

def visualize_cdf_theta1_theta2(model, device, obstacles, resolution=200, batch_size=1000, num_bubbles=10, save_path='cdf_theta1_theta2_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    
    # Generate angles for theta1 and theta2
    theta1_range = np.linspace(-np.pi, np.pi, resolution)
    theta2_range = np.linspace(-np.pi, np.pi, resolution)
    theta1_mesh, theta2_mesh = np.meshgrid(theta1_range, theta2_range)
    
    # Flatten the meshgrid
    configs = np.column_stack((theta1_mesh.flatten(), theta2_mesh.flatten()))
    
    
    # Process in batches
    cdf_values = []
    for i in range(0, len(configs), batch_size):
        batch_configs = configs[i:i+batch_size]
        batch_cdf_values = cdf_evaluate_model(model, batch_configs, obstacles, device)
        cdf_values.append(batch_cdf_values)
    
    cdf_values = np.concatenate(cdf_values)
    cdf_values = np.min(cdf_values, axis=1).reshape(resolution, resolution)
    
    # Plot the heatmap
    contour = ax.contourf(theta1_mesh, theta2_mesh, cdf_values, levels=20, cmap='viridis')
    contour = ax.contour(theta1_mesh, theta2_mesh, cdf_values, levels=[0.1], colors='red', linewidths=2)
    
    # Generate random points for bubbles
    random_theta1 = np.random.uniform(-np.pi + 0.2, np.pi - 0.2, num_bubbles)
    random_theta2 = np.random.uniform(-np.pi + 0.2, np.pi - 0.2, num_bubbles)
    random_points = np.column_stack((random_theta1, random_theta2))
    
    # Compute CDF values for random points
    random_cdf_values = cdf_evaluate_model(model, random_points, obstacles, device)
    random_cdf_values = np.min(random_cdf_values, axis=1)
    
    # Create a clip path
    clip_rect = Rectangle((-np.pi, -np.pi), 2*np.pi, 2*np.pi, fill=False)
    ax.add_patch(clip_rect)
    
    # Plot bubbles
    for point, cdf_value in zip(random_points, random_cdf_values):
        if cdf_value > 0.05:
            circle = Circle(point, cdf_value - 0.05, fill=False, color='red', alpha=0.5, clip_path=clip_rect)
            ax.add_patch(circle)
    
    ax.set_xlabel('Theta 1 (radians)')
    ax.set_ylabel('Theta 2 (radians)')
    ax.set_title('CDF Visualization for Theta 1 and Theta 2')
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')
    
    # Set the limits explicitly
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    # plt.show()

def plan_path(cdf, initial_config: np.ndarray, goal_config: np.ndarray):
    # TODO: Implement planning algorithm using CDF
    sdf = lambda x: cdf(x)
    epsilon = 5E-2
    minimum_radius = 1E-10
    num_test_positions = 3000

    mins = -1 * np.pi * np.ones(initial_config.shape)
    mins[1:] = mins[1:] /2
    maxs = 1 * np.pi * np.ones(initial_config.shape)
    maxs[1:] = maxs[1:] / 2

    overlaps_graph, max_circles, _ = get_rapidly_exploring(
        sdf,
        epsilon,
        minimum_radius,
        num_test_positions,
        mins,
        maxs,
        initial_config.tolist(),
        end_point=goal_config,
        max_retry=1000,
        max_retry_epsilon=1000,
        max_num_iterations=num_test_positions
    )

    # start_idx = position_to_max_circle_idx(overlaps_graph, initial_config)
    # if start_idx < 0:
    #     print("repairing graph for start")
    #     overlaps_graph, start_idx = trace_toward_graph_all(overlaps_graph, sdf, epsilon, minimum_radius, initial_config)

    # end_idx = position_to_max_circle_idx(overlaps_graph, goal_config)
    # if end_idx < 0:
    #     print("repairing graph for end")
    #     overlaps_graph, end_idx = trace_toward_graph_all(overlaps_graph, sdf, epsilon, minimum_radius, goal_config)

    # overlaps_graph.to_directed()

    # epath_centre_distance = get_shortest_path(
    #     lambda from_circle, to_circle: from_circle.hausdorff_distance_to(to_circle),
    #     overlaps_graph,
    #     start_idx,
    #     end_idx,
    #     cost_name="cost",
    #     return_epath=True,
    # )

    # # print(vpath_centre_distance)

    # bps, constr_bps = edgeseq_to_traj_constraint_bezier(
    #     overlaps_graph.es[epath_centre_distance[0]], initial_config, goal_config
    # )

    # cost = bezier_cost_all(bps)

    # prob = cvxpy.Problem(cvxpy.Minimize(cost), constr_bps)

    # prob.solve(verbose=True)

    # times = np.linspace(0, 1.0, 50)
    # query = np.vstack(list(map(lambda bp: bp.query(times).value, bps)))

    # # For demonstration purposes, create a simple linear interpolation between initial and goal configs
    # # n_steps = 100
    # # planned_path = overlaps_graph.vs[vpath_centre_distance[0]]["position"]
    # planned_path = query
  
    return None, max_circles # For now, return the planned_path 

if __name__ == "__main__":
    __spec__ = None
    trained_model_path = "trained_models/cdf_models/cdf_model_zeroconfigs_2_links_best.pt"  # Adjust path as needed
    torch_model = load_learned_cdf(trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(12345)

    # Create obstacles
    obstacles = create_obstacles(rng=rng)

    cdf = CDF(trained_model_path, obstacles, device)


    # Visualize SDF for theta1 and theta2
    visualize_simple_sdf_theta1_theta2(obstacles)

    # Visualize CDF for theta1 and theta2 with bubbles
    visualize_cdf_theta1_theta2(cdf.model, cdf.device, cdf.obstacle_points, num_bubbles=100)


    # Set initial and goal configurations
    initial_config = np.array([0, 0])
    goal_config = np.array([0.5, -1])

    sdf = lambda x: cdf(x)
    epsilon = 5E-2
    minimum_radius = 1E-10
    num_test_positions = 3000

    mins = -1 * np.pi * np.ones(initial_config.shape)
    mins[1:] = mins[1:] /2
    maxs = 1 * np.pi * np.ones(initial_config.shape)
    maxs[1:] = maxs[1:] / 2

    overlaps_graph, max_circles, _ = get_rapidly_exploring(
        sdf,
        epsilon,
        minimum_radius,
        num_test_positions,
        mins,
        maxs,
        initial_config,
        end_point=goal_config,
        max_retry=1000,
        max_retry_epsilon=1000,
        max_num_iterations=num_test_positions
    )


    start_idx = position_to_max_circle_idx(overlaps_graph, initial_config)

    samples_per_dim = 100
    thetas = np.meshgrid(
        *( (np.linspace(-np.pi/2, np.pi/2, samples_per_dim),) * initial_config.shape[-1] )
    )
    
    thetas_flattened = np.stack(
        tuple(map(lambda x: x.flatten(), thetas)),
        axis=-1
    )

    cdf_value, _ = cdf_evaluate_model(cdf.model, config_to_cdf_input(thetas_flattened), cdf.obstacle_points, cdf.device)

    fig_env, ax_env = plt.subplots()

    plot_environment(obstacles, initial_config, ax=ax_env)

    fig, ax = plt.subplots()

    circle_patches = [
        mpl.patches.Circle(
            circle.centre,
            circle.radius,
            fill=False
        )
        for circle in max_circles
    ]

    contour = ax.contourf(thetas[0], thetas[1], np.min(cdf_value, axis=0).reshape(thetas[0].shape), vmin=0., vmax = np.pi/2)

    cbar = fig.colorbar(contour)

    for circle in circle_patches:
        ax.add_artist(circle)

    ax.plot(initial_config[0], initial_config[1], 'rx')
    ax.plot(goal_config[0], goal_config[1], 'rd')



    # Plan path
    # planned_path, max_circles = plan_path(cdf, initial_config, goal_config)

    # Visualize the environment with the initial arm configuration
    # animate_path(obstacles, planned_path)


    print(f"Initial configuration: {initial_config}")
    print(f"Goal configuration: {goal_config}")
    # print(f"Planned path shape: {planned_path.shape}")
    print("Animation saved as 'robot_arm_animation.mp4'")
