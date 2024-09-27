import numpy as np
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle, Rectangle

from utils_env import visualize_sdf_theta1_theta2
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



def config_to_cdf_input(angles: jnp.ndarray) -> jnp.ndarray:
    """
    Convert arm configuration angles to the input format required by cdf_evaluate_model.
    
    Args:
    angles (jnp.ndarray): Array of joint angles.
    
    Returns:
    jnp.ndarray: Formatted input for cdf_evaluate_model.
    """
    num_angles = angles.shape[0]
    return jnp.concatenate([
        angles,
        jnp.sin(angles),
        jnp.cos(angles)
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


def visualize_cdf_theta1_theta2(jax_params, obstacles, resolution=100, batch_size=1000, num_bubbles=10, save_path='cdf_theta1_theta2_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)  # Adjusted figure size to be rectangular
    
    # Generate angles for theta1 and theta2
    theta1_range = jnp.linspace(-jnp.pi, jnp.pi, resolution)
    theta2_range = jnp.linspace(-jnp.pi/2, jnp.pi/2, resolution)
    theta1_mesh, theta2_mesh = jnp.meshgrid(theta1_range, theta2_range)
    
    # Flatten the meshgrid for batched computation
    theta1_flat = theta1_mesh.flatten()
    theta2_flat = theta2_mesh.flatten()
    
    # Create angles array with only two angles
    angles_flat = jnp.column_stack((theta1_flat, theta2_flat))
    
    # Combine all obstacle points
    all_obstacle_points = jnp.concatenate(obstacles, axis=0)
    
    # Batched CDF computation
    @jax.vmap
    def compute_cdf(angles):
        config = config_to_cdf_input(angles)
        cdf_values, _ = cdf_evaluate_model(jax_params, config, all_obstacle_points)
        return jnp.min(cdf_values)
    
    # Process in batches
    cdf_values = []
    for i in range(0, len(angles_flat), batch_size):
        batch = angles_flat[i:i+batch_size]
        cdf_values.append(compute_cdf(batch))
    
    cdf_values = jnp.concatenate(cdf_values)
    cdf_values = cdf_values.reshape(resolution, resolution)
    
    # Plot the heatmap
    heatmap = ax.imshow(cdf_values, extent=[-jnp.pi, jnp.pi, -jnp.pi/2, jnp.pi/2], origin='lower', 
                        aspect='auto', cmap='viridis')
    #contour = ax.contour(theta1_mesh, theta2_mesh, cdf_values, levels=[0.1], colors='red', linewidths=2)
    
    # Generate random points for bubbles
    random_theta1 = np.random.uniform(-np.pi, np.pi, num_bubbles)
    random_theta2 = np.random.uniform(-np.pi/2, np.pi/2, num_bubbles)
    random_points = jnp.column_stack((random_theta1, random_theta2))
    
    # Compute CDF values for random points
    random_cdf_values = compute_cdf(random_points)
    
    # Create a clip path
    clip_rect = Rectangle((-np.pi, -np.pi/2), 2*np.pi, np.pi, fill=False)
    ax.add_patch(clip_rect)
    
    # Plot bubbles
    for point, cdf_value in zip(random_points, random_cdf_values):
        if cdf_value > 0.1:
            circle = Circle(point, cdf_value - 0.1, fill=False, color='red', alpha=0.5, clip_path=clip_rect)
            ax.add_patch(circle)
    
    ax.set_xlabel('Theta 1 (radians)')
    ax.set_ylabel('Theta 2 (radians)')
    ax.set_title('CDF Visualization with Bubbles for Theta 1 and Theta 2')
    
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Minimum CDF Value')
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')
    
    # Set the limits explicitly
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi/2, np.pi/2)
    
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
    trained_model_path = "trained_models/cdf_models/cdf_model_4_256_2_links.pt"  # Adjust path as needed
    jax_net, jax_params = load_learned_cdf(trained_model_path)

    # Create obstacles
    obstacles = create_obstacles()


    # Set initial and goal configurations
    initial_config = np.array([0, 0])
    goal_config = np.array([np.pi/2, np.pi/4])


    # Check CDF values for specific obstacle points
    test_points = np.array([[4, 1], [2, 1], [-4, 1]])
    config_input = config_to_cdf_input(initial_config)
    cdf_values, _ = cdf_evaluate_model(jax_params, config_input, test_points)
    
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
    visualize_sdf_theta1_theta2(params_list, obstacles)

    # Visualize CDF for theta1 and theta2 with bubbles
    visualize_cdf_theta1_theta2(jax_params, obstacles, num_bubbles=50)

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