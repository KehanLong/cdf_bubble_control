import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.arm_2d_config import shapes, NUM_LINKS, end_points
import jax.numpy as jnp

import jax
from utils.sdf_net import SDFNet

from utils.config import HIDDEN_SIZE



@jax.jit
def evaluate_model(params, points):
    def apply_model(params, points):
        return SDFNet(HIDDEN_SIZE, 4).apply(params, points)
    
    outputs = apply_model(params, points)
    
    grad_fn = jax.grad(lambda x: apply_model(params, x).sum())
    gradients = jax.vmap(grad_fn)(points)

    return outputs, gradients

@jax.jit
def rotate_point(point, angle):
    x, y = point
    return jnp.array([
        x * jnp.cos(angle) - y * jnp.sin(angle),
        x * jnp.sin(angle) + y * jnp.cos(angle)
    ])



@jax.jit
def transform_shape(shape, angle, translation):
    shape_array = jnp.array(shape)
    cos_angle, sin_angle = jnp.cos(angle), jnp.sin(angle)
    rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated = jnp.dot(shape_array, rotation_matrix.T)
    return rotated + jnp.array(translation)

@jax.jit
def transform_points_to_local(points, angle, translation):
    """Transform points from global frame to local frame of a link."""
    # First, translate the points
    translated_points = points[:, :2] - jnp.array(translation)
    
    # Then, rotate the points
    cos_angle, sin_angle = jnp.cos(-angle), jnp.sin(-angle)
    rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_points = jnp.dot(translated_points, rotation_matrix.T)
    
    # Add back the z-coordinate (which is always 0 in our 2D case)
    return jnp.column_stack((rotated_points, jnp.zeros(len(points))))



@jax.jit
def forward_kinematics(angles):
    def body_fn(i, carry):
        current_pos, current_angle, joint_positions = carry
        current_angle += angles[i]
        end_point = jnp.array(end_points)[i]  # Convert end_points to a JAX array and index it
        rotated_end = rotate_point(end_point, current_angle)
        current_pos = current_pos + rotated_end
        joint_positions = joint_positions.at[i + 1].set(current_pos)
        return current_pos, current_angle, joint_positions

    joint_positions = jnp.zeros((NUM_LINKS + 1, 2))
    carry = (jnp.array([0.0, 0.0]), 0.0, joint_positions)
    _, _, joint_positions = jax.lax.fori_loop(0, NUM_LINKS, body_fn, carry)
    return joint_positions


@jax.jit
def calculate_arm_sdf(points, angles, params_list):
    """
    Calculate the signed distance from points to the robot arm using JAX.
    Args:
        points (array): (N, 2) array of (x, y) coordinates of points in the workspace,
                        or a single (2,) array for one point
        angles (list): List of joint angles
        params_list (list): List of learned network parameters for each link
    Returns:
        array: Minimum signed distances from the points to the whole robot arm
    """
    # Ensure points is always a 2D array
    points = jnp.atleast_2d(points)
    
    joint_positions = forward_kinematics(angles)
    points_3d = jnp.pad(points, ((0, 0), (0, 1)))  # Convert to 3D points
    min_distances = jnp.full(points.shape[0], jnp.inf)
    current_angle = 0.0

    for i in range(NUM_LINKS):
        if i < len(angles):
            current_angle += angles[i]

        # Transform points to local frame of the current link
        local_points = transform_points_to_local(points_3d, current_angle, joint_positions[i])

        # Get the correct params for the current link
        params = params_list[i]

        # Evaluate the SDF model for the link
        distances, _ = evaluate_model(params, local_points)

        # Update minimum distances
        min_distances = jnp.minimum(min_distances, distances.squeeze())

    # If input was a single point, return a scalar
    return min_distances[0] if points.shape[0] == 1 else min_distances

@jax.jit
def calculate_arm_sdf_with_grad(point, angles, params_list):
    def sdf_func(angles):
        return calculate_arm_sdf(point, angles, params_list)
    
    sdf_value, gradient = jax.value_and_grad(sdf_func)(angles)
    return sdf_value, gradient

def visualize_arm_sdf(angles, params_list, save_path='arm_sdf_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Generate points for SDF evaluation
    n_points = 200
    x = np.linspace(-20, 20, n_points)
    y = np.linspace(-20, 20, n_points)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx.flatten(), yy.flatten()), axis=-1)
    
    # Calculate SDF for all points at once
    distances = calculate_arm_sdf(points, angles, params_list)
    distances = distances.reshape(n_points, n_points)

    heatmap = ax.imshow(distances, cmap='coolwarm', extent=[-20, 20, -20, 20], origin='lower', aspect='equal', vmin=-2, vmax=15)
    contour = ax.contour(xx, yy, distances, levels=[0], colors='black', linewidths=1)

    joint_positions = forward_kinematics(angles)
    current_angle = 0
    
    for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
        if i < len(angles):
            current_angle += angles[i]
        transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
        ax.fill(*zip(*transformed_shape), alpha=0.5)
        ax.plot(joint_pos[0], joint_pos[1], 'bo', markersize=8)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title('Combined SDF for Robot Arm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Signed Distance')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    # plt.show()

def test_calculate_arm_sdf_with_grad(params_list):
    print("Testing calculate_arm_sdf_with_grad function:")

    # Test cases
    test_cases = [
        (jnp.array([-8, 4.0]), jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])),
        (jnp.array([5.0, 5.0]), jnp.array([np.pi/4, -np.pi/4, 0.0, np.pi/4, -np.pi/4])),
        (jnp.array([-8.0, 4.0]), jnp.array([np.pi/2, 0.0, -np.pi/4, np.pi/3, 0.0])),
    ]

    for i, (point, angles) in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"Point: {point}, Angles: {angles}")

        # Calculate SDF and gradient
        sdf_value, gradient = calculate_arm_sdf_with_grad(point, angles, params_list)

        print(f"SDF value: {sdf_value}")
        print(f"Gradient: {gradient}")

        # Verify gradient using finite differences
        epsilon = 1e-3
        numerical_gradient = []

        for j in range(len(angles)):
            angles_plus = angles.at[j].add(epsilon)
            angles_minus = angles.at[j].add(-epsilon)

            sdf_plus = calculate_arm_sdf(point, angles_plus, params_list)
            sdf_minus = calculate_arm_sdf(point, angles_minus, params_list)

            numerical_grad = (sdf_plus - sdf_minus) / (2 * epsilon)
            numerical_gradient.append(numerical_grad)

        numerical_gradient = np.array(numerical_gradient)
        print(f"Numerical gradient: {numerical_gradient}")

        # Compare analytical and numerical gradients
        gradient_diff = np.abs(gradient - numerical_gradient)
        print(f"Max difference between analytical and numerical gradients: {np.max(gradient_diff)}")

        if np.allclose(gradient, numerical_gradient, atol=1e-3):
            print("Gradient check passed!")
        else:
            print("Gradient check failed. The implementation might be incorrect.")

def main():
    angles = jnp.array([np.pi/2, -np.pi/4, 0, np.pi/3 ,-np.pi/4])
    
    params_list = []
    for i in range(NUM_LINKS):
        params = jnp.load(f"trained_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)
    
    visualize_arm_sdf(angles, params_list)

    # Add this line to run the test function
    # test_calculate_arm_sdf_with_grad(params_list)

if __name__ == "__main__":
    main()