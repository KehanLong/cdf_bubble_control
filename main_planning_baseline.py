import numpy as np
from typing import List
from utils_env import create_obstacles, plot_environment
from cdf_evaluate_jax import load_learned_cdf_to_jax, cdf_evaluate_model_jax
import torch
import matplotlib.pyplot as plt
import imageio
import jax
import jax.numpy as jnp

from control_and_planning.mppi_functional import setup_mppi_controller
from utils_new import forward_kinematics

class CDF:
    def __init__(self, weights_file, obstacles):
        self.model, self.jax_params = load_learned_cdf_to_jax(weights_file)
        self.obstacle_points = concatenate_obstacle_list(obstacles)
    
    def __call__(self, configurations):
        cdf_values = cdf_evaluate_model_jax(self.jax_params, configurations, self.obstacle_points)
        return np.min(cdf_values, axis=1)

def robot_dynamics_step(state, input, dt=0.05):
    # Assuming state and input are joint angles and velocities
    joint_angles = state
    joint_velocities = input

    # Update joint angles based on velocities
    joint_angles_next = joint_angles + joint_velocities * dt

    return joint_angles_next

def animate_path(obstacles: List[np.ndarray], actual_path: np.ndarray, selected_trajectories: List[np.ndarray], goal_position: np.ndarray, fps: int = 20, duration: float = 10.0):
    """
    Create an animation of the robot arm moving along the actual path and showing selected trajectories.
    
    Args:
    obstacles (List[np.ndarray]): List of obstacle arrays.
    actual_path (np.ndarray): Array of shape (n_steps, robot_n) containing the actual configurations.
    selected_trajectories (List[np.ndarray]): List of arrays, each of shape (horizon, robot_n) containing selected trajectories.
    goal_position (np.ndarray): The goal position for the end effector.
    fps (int): Frames per second for the animation.
    duration (float): Duration of the animation in seconds.
    
    Returns:
    None: Saves the animation as a video file.
    """
    n_frames = int(fps * duration)
    path_indices = np.linspace(0, len(actual_path) - 1, n_frames, dtype=int)
    
    frames = []
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in path_indices:
        ax.clear()
        config = actual_path[i]


        plot_environment(obstacles, config, ax=ax)
        
        # Plot selected trajectory for the current step
        if i < len(selected_trajectories):
            selected_ee_positions = np.array([forward_kinematics(c) for c in selected_trajectories[i]])
            ax.plot(selected_ee_positions[:, 0], selected_ee_positions[:, 1], 'r--', label='Selected Trajectory')
        
        # Plot the goal position as a star
        ax.plot(goal_position[0], goal_position[1], 'y*', markersize=15, label='Goal')
        
        ax.legend()
        ax.set_title(f"Step {i+1}/{len(actual_path)}")
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    # Save as video using imageio
    imageio.mimsave('robot_arm_animation.mp4', frames, fps=fps)
    print("Animation saved as 'robot_arm_animation.mp4'")

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

    # plt.show()

def plan_path(obstacles: List[np.ndarray], initial_config: np.ndarray, goal_position: np.ndarray, jax_params):
    # Set up the MPPI controller

    horizon = 20
    cost_goal_coeff = 8.0
    cost_safety_coeff = 1.0
    cost_perturbation_coeff = 0.1
    cost_goal_coeff_final = 8.0
    cost_safety_coeff_final = 1.0
    mppi_controller = setup_mppi_controller(
        learned_CDF=jax_params,
        robot_n=initial_config.shape[0],
        input_size=initial_config.shape[0],
        initial_horizon=horizon,
        samples=2000,
        control_bound=0.4,
        dt=0.05,
        use_GPU=True, 
        cost_goal_coeff=cost_goal_coeff,
        cost_safety_coeff=cost_safety_coeff,
        cost_perturbation_coeff=cost_perturbation_coeff,
        cost_goal_coeff_final=cost_goal_coeff_final,
        cost_safety_coeff_final=cost_safety_coeff_final
    )

    # Initialize state and control sequence
    current_state = initial_config
    actual_path = []
    selected_trajectories = []
    safety_margin = 0.05

    # Initialize random key and control sequence U
    key = jax.random.PRNGKey(0)  # You can use any seed value
    U = jnp.zeros((horizon, initial_config.shape[0]))  # Assuming horizon is 10

    step = 0
    # Iterate until the end effector reaches the goal position
    while np.linalg.norm(forward_kinematics(current_state) - goal_position) > 0.1:  # Example threshold

        print('distance to goal: ', np.linalg.norm(forward_kinematics(current_state) - goal_position), 'step: ', step)
        #print(current_state)
        key, subkey = jax.random.split(key)


        cdf_values, _ = cdf_evaluate_model_jax(jax_params, current_state, concatenate_obstacle_list(obstacles))
        min_cdf_value = jnp.min(cdf_values)
        print('min cdf value: ', min_cdf_value)
        
        # Generate control inputs using MPPI
        _, selected_states, action, U = mppi_controller(
            key=subkey,
            U=U,
            init_state=current_state,
            goal=goal_position,
            obstaclesX=concatenate_obstacle_list(obstacles),
            safety_margin=safety_margin
        )

        # Update the current state using the first action
        current_state = robot_dynamics_step(current_state, action.flatten())

        actual_path.append(current_state)
        selected_trajectories.append(selected_states.T)  # Store the entire selected trajectory

        step += 1
        if step > 200:  # Ad
            break

    return np.array(actual_path), selected_trajectories

if __name__ == "__main__":
    trained_model_path = "trained_models/cdf_models/cdf_model_2_links.pt"  # Adjust path as needed, 2_links or 4_links
    jax_model, jax_params = load_learned_cdf_to_jax(trained_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(12345)

    # Create obstacles
    obstacles = create_obstacles(num_points=25, rng=rng)

    cdf = CDF(trained_model_path, obstacles)

    # Set up initial configuration and goal position for the end effector
    initial_config = np.array([0.0, 0.0])  # Initial joint configuration
    goal_position = np.array([-2., 2.])  # Goal position for the end effector

    # Plan path
    actual_path, selected_trajectories = plan_path(obstacles, initial_config, goal_position, cdf.jax_params)

    # Animate path
    animate_path(obstacles, actual_path, selected_trajectories, goal_position)
