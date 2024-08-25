import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from control.mppi_functional import setup_mppi_controller
import imageio
import time

from utils_new import *


def visualize_env_with_robot(goal_pos, obstacle_centers, obstacle_lengths, obstacle_widths, obstacle_orientations, obstacle_points,
                             robot_state, robot_length, robot_width, ax):

    # Plot the goal position
    ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')

    # Plot the obstacle rectangles
    for center, length, width, orientation in zip(obstacle_centers, obstacle_lengths, obstacle_widths, obstacle_orientations):
    
        corner_x = center[0] - length/2 * np.cos(orientation) + width/2 * np.sin(orientation)
        corner_y = center[1] - length/2 * np.sin(orientation) - width/2 * np.cos(orientation)

        rect = Rectangle(xy=(corner_x, corner_y), width=length, height=width,
                         angle=np.rad2deg(orientation), linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    # Plot the obstacle point clouds
    ax.plot(obstacle_points[:, 0], obstacle_points[:, 1], 'r.', markersize=5, label='Obstacle Points')

    # Compute the bottom-left corner of the rotated robot rectangle
    robot_center_x, robot_center_y, robot_angle = robot_state
    corner_x = robot_center_x - robot_length/2 * np.cos(robot_angle) + robot_width/2 * np.sin(robot_angle)
    corner_y = robot_center_y - robot_length/2 * np.sin(robot_angle) - robot_width/2 * np.cos(robot_angle)

    # Plot the robot as a rectangle
    robot_rect = Rectangle((corner_x, corner_y), robot_length, robot_width,
                           angle=np.rad2deg(robot_angle), color='b', alpha=0.2)
    ax.add_patch(robot_rect)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()

    return fig, ax



# Set up the MPPI controller
num_samples = 2000
costs_lambda = 0.02
cost_goal_coeff = 15.0
cost_safety_coeff = 1.0
cost_goal_coeff_final = 18.0

cost_safety_coeff_final = 1.1
control_bound = 3.0
use_GPU = True
prediction_horizon = 30
dt = 0.05
U = 0.0 * jnp.ones((prediction_horizon, 2))

trained_params = jnp.load(f"trained_models/trailer_model_4_16.npy", allow_pickle=True).item()

mppi = setup_mppi_controller(learned_CSDF = trained_params, robot_n=3, initial_horizon=prediction_horizon,
                             samples=num_samples, input_size=2, control_bound=control_bound,
                             dt=dt, u_guess=U, use_GPU=use_GPU, costs_lambda=costs_lambda,
                             cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff,
                             cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final)


goal_position = np.array([5.0, 5.0])
obstacle_centers = np.array([[4.2, 5.0], [5.8, 5.0], [0.0, 3.0]])  # Added a new obstacle at [0, 3]
obstacle_lengths = np.array([1.5, 1.5, 1.5])  # Added length for the new obstacle
obstacle_widths = np.array([0.5, 0.5, 0.5])  # Added width for the new obstacle
obstacle_orientations = np.array([np.pi/2, np.pi/2, 0.0])  # Added orientation for the new obstacle

moving_obstacle_velocity = 0.0  # Velocity of the moving obstacle

safety_margin = 0.05 

robot_length = 1.5
robot_width = 0.5

# Initialize the robot state
robot_init_state = jnp.array([5.0, 3.0, 0.0])  # [x, y, theta]
robot_state = robot_init_state

# Simulate the robot motion
num_steps = 100
robot_trajectory = [robot_state]
frames = []


fig, ax = plt.subplots(figsize=(8, 8))

for step in range(num_steps):
    ax.clear()

    key = jax.random.PRNGKey(step)

    # Update the position of the moving obstacle
    obstacle_centers[-1, 0] += moving_obstacle_velocity * dt

    # Generate the updated obstacle point clouds
    goal_pos, obstacles = parking_env_generate(goal_position, obstacle_centers, obstacle_lengths, obstacle_widths,
                                               obstacle_orientations, points_per_unit=10)

    # Transform obstacle points to robot frame
    sdf_distances, _ = evaluate_model(trained_params, obstacles, robot_state)

    start_time = time.time()

    sampled_states, states_final, action, U = mppi(key, U, robot_state, goal_position, obstacles, safety_margin)

    print('MPPI solver time:', time.time() - start_time)

    robot_state = robot_dynamics_step(robot_state, action, dt)
    robot_trajectory.append(np.ravel(robot_state))

    robot_traj = np.array(robot_trajectory)

    visualize_env_with_robot(goal_pos, obstacle_centers, obstacle_lengths, obstacle_widths,
                             obstacle_orientations, obstacles, robot_state, robot_length, robot_width, ax)

    ax.plot(states_final[0, :], states_final[1, :], 'b-')

    # Plot all sampled states
    for i in range(10):
        sample_trajectory = sampled_states.reshape(num_samples, 3, prediction_horizon)
        ax.plot(sample_trajectory[i, 0, :], sample_trajectory[i, 1, :], 'g.', markersize=1)

    ax.set_title(f'Step {step+1}')

    plt.pause(0.01)  # Add a small pause to allow the plot to update

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

    # Check if the robot is close enough to the goal
    if np.linalg.norm(robot_state[:2] - goal_position) < 0.1 and robot_state[2] - np.pi/2 < 0.01:
        print("Goal reached!")
        # Append the last frame for 0.5 seconds
        last_frame = frames[-1]
        for _ in range(int(0.5 / dt)):
            frames.append(last_frame)
        break

plt.close(fig)

# Save the frames as an MP4 file
imageio.mimsave('robot_trajectory.mp4', frames, fps=int(1/dt))