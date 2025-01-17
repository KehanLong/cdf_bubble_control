import numpy as np
import jax.numpy as jnp
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate import load_learned_cdf, cdf_evaluate_model
import torch
import matplotlib.pyplot as plt

from utils_env import visualize_simple_sdf_theta1_theta2




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


def visualize_cdf_for_joint_pair(ax, model, device, joint_pair, num_links, obstacles, resolution=200, batch_size=1000):
    theta = np.linspace(-np.pi, np.pi, resolution)
    Theta1, Theta2 = np.meshgrid(theta, theta)
    
    configs = np.zeros((resolution * resolution, num_links))
    configs[:, joint_pair[0]] = Theta1.flatten()
    configs[:, joint_pair[1]] = Theta2.flatten()
    
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
    contour = ax.contourf(Theta1, Theta2, cdf_values, levels=20, cmap='viridis')
    #zero_level = ax.contour(Theta1, Theta2, cdf_values, levels=[0.01], colors='red', linewidths=2)
    #lt.colorbar(contour, ax=ax, label='CDF Value')

    # Add labels to the zero level set
    #ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    ax.set_xlabel(f'θ{joint_pair[0] + 1} (radians)')
    ax.set_ylabel(f'θ{joint_pair[1] + 1} (radians)')
    ax.set_title(f'CDF Visualization for Joint Pair {joint_pair}')

    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True)


def main():
    # Load the CDF model
    trained_model_path = "trained_models/cdf_models/cdf_model_2_links_truncated_new.pt"  # Adjust path as needed
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



    #
    plot_environment(obstacles, initial_config, save_path='high_res_environment.png')


    
    # Visualize SDF for theta1 and theta2
    # visualize_simple_sdf_theta1_theta2(obstacles)

    num_links = initial_config.shape[0]
    joint_pairs = [(i, j) for i in range(num_links) for j in range(i + 1, num_links)]
    num_plots = len(joint_pairs)

    # Calculate grid dimensions
    cols = 2
    rows = (num_plots + cols - 1) // cols  # Ceiling division to determine rows

    # Set figsize to ensure square subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axs = axs.ravel()

    for i, joint_pair in enumerate(joint_pairs):
        visualize_cdf_for_joint_pair(axs[i], torch_model, device, joint_pair, num_links, obstacles)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig('cdf_joint_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("CDF joint pairs plot saved as 'cdf_joint_pairs.png'")


if __name__ == "__main__":
    main()
