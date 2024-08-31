import numpy as np
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio

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
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    # Save as video using imageio
    imageio.mimsave('robot_arm_animation.mp4', frames, fps=fps)



def config_to_cdf_input(angles: jnp.ndarray) -> jnp.ndarray:
    """
    Convert arm configuration angles to the input format required by cdf_evaluate_model.
    
    Args:
    angles (jnp.ndarray): Array of 5 joint angles.
    
    Returns:
    jnp.ndarray: Formatted input for cdf_evaluate_model.
    """
    return jnp.array([
        angles[0], angles[1], angles[2], angles[3], angles[4],
        jnp.sin(angles[0]), jnp.sin(angles[1]), jnp.sin(angles[2]), jnp.sin(angles[3]), jnp.sin(angles[4]),
        jnp.cos(angles[0]), jnp.cos(angles[1]), jnp.cos(angles[2]), jnp.cos(angles[3]), jnp.cos(angles[4])
    ])

def concatenate_obstacle_list(obstacle_list):
    """
    Concatenate a list of obstacle arrays into a single numpy array.
    
    Args:
    obstacle_list (list): List of numpy arrays, each of shape (M, 2)
    
    Returns:
    np.ndarray: A single numpy array of shape (N, 2) where N is the total number of points
    """
    return np.concatenate(obstacle_list, axis=0)

def plan_path(obstacles: List[np.ndarray], initial_config: np.ndarray, goal_config: np.ndarray, jax_params):
    # TODO: Implement planning algorithm using CDF

    config = config_to_cdf_input(initial_config)
    obstacle_points = concatenate_obstacle_list(obstacles)
    cdf_values, _ = cdf_evaluate_model(jax_params, config, obstacle_points)
    print('estimated bubble radius: ', min(cdf_values))


    print("Path planning not yet implemented.")

    # For demonstration purposes, create a simple linear interpolation between initial and goal configs
    n_steps = 100
    planned_path = np.linspace(initial_config, goal_config, n_steps)
  
    return planned_path  # For now, return the planned_path 

def main():
    # Load the CDF model
    trained_model_path = "trained_models/cdf_models/cdf_model_5_256_with_surface.pt"  # Adjust path as needed
    jax_net, jax_params = load_learned_cdf(trained_model_path)

    # Create obstacles
    obstacles = create_obstacles()

    # Set initial and goal configurations
    initial_config = np.array([0, 0, 0, 0, 0])
    goal_config = np.array([np.pi/2, np.pi/4, -np.pi/4, 0, 0])


    # Check CDF values for specific obstacle points
    test_points = np.array([[4, 1], [2, 1], [10, 1]])
    config_input = config_to_cdf_input(initial_config)
    cdf_values, _ = cdf_evaluate_model(jax_params, config_input, test_points)
    
    print("CDF values for specific obstacle points:")
    for point, cdf_value in zip(test_points, cdf_values):
        print(f"Point {point}: CDF value = {cdf_value}")


    # Plan path
    planned_path = plan_path(obstacles, initial_config, goal_config, jax_params)

    # Visualize the environment with the initial arm configuration
    animate_path(obstacles, planned_path)


    print(f"Initial configuration: {initial_config}")
    print(f"Goal configuration: {goal_config}")
    print(f"Planned path shape: {planned_path.shape}")
    print("Animation saved as 'robot_arm_animation.mp4'")


if __name__ == "__main__":
    main()