import numpy as np
import jax.numpy as jnp
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import torch
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle, Rectangle

from utils_env import visualize_sdf_theta1_theta2, visualize_simple_sdf_theta1_theta2
from data.arm_2d_config import NUM_LINKS

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



def config_to_cdf_input(angles: np.ndarray) -> np.ndarray:
    """
    Convert arm configuration angles to the input format required by cdf_evaluate_model.
    
    Args:
    angles (np.ndarray): Array of joint angles.
    
    Returns:
    np.ndarray: Formatted input for cdf_evaluate_model.
    """
    num_angles = angles.shape[0]
    return np.concatenate([
        angles,
        np.sin(angles),
        np.cos(angles)
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


def visualize_cdf_theta1_theta2(model, device, obstacles, resolution=200, batch_size=1000, num_bubbles=10, save_path='cdf_theta1_theta2_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    
    # Generate angles for theta1 and theta2
    theta1_range = np.linspace(-np.pi, np.pi, resolution)
    theta2_range = np.linspace(-np.pi, np.pi, resolution)
    theta1_mesh, theta2_mesh = np.meshgrid(theta1_range, theta2_range)
    
    # Flatten the meshgrid
    configs = np.column_stack((theta1_mesh.flatten(), theta2_mesh.flatten()))
    
    # Combine all obstacle points
    all_obstacle_points = concatenate_obstacle_list(obstacles)
    
    # Process in batches
    cdf_values = []
    for i in range(0, len(configs), batch_size):
        batch_configs = configs[i:i+batch_size]
        batch_cdf_values = cdf_evaluate_model(model, batch_configs, all_obstacle_points, device)
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
    random_cdf_values = cdf_evaluate_model(model, random_points, all_obstacle_points, device)
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
    trained_model_path = "trained_models/cdf_models/cdf_model_zeroconfigs_2_links_best.pt"  # Adjust path as needed
    torch_model = load_learned_cdf(trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create obstacles
    obstacles = create_obstacles()


    # Set initial and goal configurations
    initial_config = np.array([0, 0])
    goal_config = np.array([np.pi/2, np.pi/4])


    # Check CDF values for specific obstacle points
    test_points = np.array([[2, 0], [1, 0], [-2, 1]])
    cdf_values = cdf_evaluate_model(torch_model, initial_config, test_points, device)
    
    print("CDF values for specific obstacle points:")
    for point, cdf_value in zip(test_points, cdf_values):
        print(f"Point {point}: CDF value = {cdf_value}")


    params_list = []
    for i in range(NUM_LINKS):
        params = jnp.load(f"trained_models/sdf_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)

    #
    plot_environment(obstacles, initial_config, save_path='high_res_environment.png')


    
    # Visualize SDF for theta1 and theta2
    # visualize_simple_sdf_theta1_theta2(obstacles)

    # Visualize CDF for theta1 and theta2 with bubbles
    visualize_cdf_theta1_theta2(torch_model, device, obstacles, num_bubbles=100)

    # Plan path
    # planned_path = plan_path(obstacles, initial_config, goal_config, jax_params)

    # # Visualize the environment with the initial arm configuration
    # animate_path(obstacles, planned_path)


    # print(f"Initial configuration: {initial_config}")
    # print(f"Goal configuration: {goal_config}")
    # print(f"Planned path shape: {planned_path.shape}")
    # print("Animation saved as 'robot_arm_animation.mp4'")


if __name__ == "__main__":
    main()
