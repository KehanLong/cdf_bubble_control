import jax
import numpy as np
import jax.numpy as jnp
from utils.sdf_net import SDFNet
from utils.config import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@jax.jit
def evaluate_model(params, points, robot_state):
    # Extract robot position and orientation
    robot_x, robot_y, robot_theta = robot_state

    # Create rotation matrix
    cos_theta = jnp.cos(robot_theta)
    sin_theta = jnp.sin(robot_theta)
    rotation_matrix = jnp.array([[cos_theta, sin_theta],
                                 [-sin_theta, cos_theta]])

    # Translate and rotate obstacle points
    translated_points = points - jnp.array([robot_x, robot_y]).squeeze()
    transformed_points = jnp.dot(translated_points, rotation_matrix).squeeze()

    # Append a column of zeros to make points 3D
    transformed_points = jnp.hstack((transformed_points, jnp.zeros((transformed_points.shape[0], 1))))

    # Predict signed distances
    def apply_model(params, points):
        return SDFNet(HIDDEN_SIZE, 4).apply(params, points)

    outputs = apply_model(params, transformed_points)

    # Compute gradients
    grad_fn = jax.grad(lambda x: apply_model(params, x).sum())
    gradients = jax.vmap(grad_fn)(transformed_points)

    return outputs, gradients



def sample_points_on_circle(center, radius, num_points):

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    points = np.vstack((x, y)).T
    return points

def generate_obstacle_point_cloud(obstacles, radius, num_points_per_obstacle):

    point_cloud = []
    for center in obstacles:
        points = sample_points_on_circle(center, radius, num_points_per_obstacle)
        point_cloud.append(points)
    
    return np.vstack(point_cloud)


def robot_dynamics_step(state, input, dt):
    # Unicycle robot dynamics
    x, y, theta = state
    v, omega = input
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    return np.array([x_next, y_next, theta_next])


def sample_rectangle_boundary(center, length, width, orientation, points_per_unit):
    # Determine the number of points based on the length and width
    num_points_length = int(length * points_per_unit)
    num_points_width = int(width * points_per_unit)

    # Create points on the edges of the rectangle
    x_left = np.full(num_points_width, -length / 2)
    x_right = np.full(num_points_width, length / 2)
    y_bottom = np.linspace(-width / 2, width / 2, num_points_width)
    y_top = np.linspace(-width / 2, width / 2, num_points_width)

    x_bottom = np.linspace(-length / 2, length / 2, num_points_length)
    x_top = np.linspace(-length / 2, length / 2, num_points_length)
    y_left = np.full(num_points_length, -width / 2)
    y_right = np.full(num_points_length, width / 2)

    # Concatenate the points
    x = np.concatenate((x_left, x_right, x_bottom, x_top))
    y = np.concatenate((y_bottom, y_top, y_left, y_right))
    points = np.vstack((x, y)).T

    # Create a rotation matrix based on the orientation
    cos_theta = np.cos(orientation)
    sin_theta = np.sin(orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Rotate and translate the points
    rotated_points = np.dot(points, rotation_matrix)
    translated_points = rotated_points + center

    return translated_points


def parking_env_generate(goal_pos, obstacle_centers, obstacle_lengths, obstacle_widths, obstacle_orientations, points_per_unit=5):
    # Initialize an empty list to store the point clouds
    obstacle_points = []

    # Iterate over each obstacle
    for center, length, width, orientation in zip(obstacle_centers, obstacle_lengths, obstacle_widths, obstacle_orientations):
        # Sample points on the boundary of the rectangle
        boundary_points = sample_rectangle_boundary(center, length, width, orientation, points_per_unit)
        obstacle_points.append(boundary_points)

    # Concatenate all the point clouds into a single array
    obstacle_points = np.vstack(obstacle_points)

    return goal_pos, obstacle_points

def visualize_env(goal_pos, obstacle_centers, obstacle_lengths, obstacle_widths, obstacle_orientations, obstacle_points):
    fig, ax = plt.subplots(figsize=(8, 8))

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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    goal_position = np.array([5.0, 5.0])
    obstacle_centers = np.array([[4.2, 5.0], [5.8, 5.0]])
    obstacle_lengths = np.array([1.0, 1.0])
    obstacle_widths = np.array([0.4, 0.4])
    obstacle_orientations = np.array([np.pi/2, np.pi/2])

    goal_pos, obstacle_points = parking_env_generate(goal_position, obstacle_centers, obstacle_lengths, obstacle_widths,
                                                     obstacle_orientations, points_per_unit=10)
    
    visualize_env(goal_pos, obstacle_centers, obstacle_lengths, obstacle_widths, obstacle_orientations, obstacle_points)
