import numpy as np
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from sdf_marching.samplers import get_rapidly_exploring
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.samplers.tracing import trace_toward_graph_all
from sdf_marching.discrete import get_shortest_path
from sdf_marching.cvx import edgeseq_to_traj_constraint_bezier, bezier_cost_all

import cvxpy

class CDF:
    def __init__(self, weights_file, obstacles):
        self.jax_net, self.jax_params = load_learned_cdf(weights_file)
        self.obstacle_points = concatenate_obstacle_list(obstacles)
    
    def __call__(self, configurations):
        cdf_value, gradients = cdf_evaluate_model(self.jax_params, config_to_cdf_input(configurations), self.obstacle_points)
        return np.min(
            cdf_value,
        )


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

def wrap_angles(angles: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(
        angles + np.pi,
        2 * np.pi
    ) - np.pi


def config_to_cdf_input(angles: jnp.ndarray) -> jnp.ndarray:
    """
    Convert arm configuration angles to the input format required by cdf_evaluate_model.
    
    Args:
    angles (jnp.ndarray): Array of joint angles.
    
    Returns:
    jnp.ndarray: Formatted input for cdf_evaluate_model.
    """
    num_angles = angles.shape[0]

    angles = wrap_angles(angles)

    return jnp.concatenate([
        angles,
        jnp.sin(angles),
        jnp.cos(angles)
    ], axis=-1)

def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
    obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
    np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)

def plan_path(cdf, initial_config: np.ndarray, goal_config: np.ndarray):

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

    trained_model_path = "trained_models/cdf_models/cdf_model_4_256_2_links.pt"  # Adjust path as needed

    rng = np.random.default_rng(12345)

    # Create obstacles
    obstacles = create_obstacles(rng=rng)

    # Create the abstraction
    cdf = CDF(trained_model_path, obstacles)

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

    cdf_value, _ = cdf_evaluate_model(cdf.jax_params, config_to_cdf_input(thetas_flattened), cdf.obstacle_points)

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

