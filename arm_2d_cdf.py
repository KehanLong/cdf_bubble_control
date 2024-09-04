import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.arm_2d_config import NUM_LINKS, shapes

from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import jit, vmap

import time

# Import necessary functions from arm_2d_utils.py
from arm_2d_utils import (
    calculate_arm_sdf,
    calculate_arm_sdf_with_grad,
    forward_kinematics,
    transform_shape
)

@jit
def batch_objective_function(angles_batch, point, params_list):
    """
    Batch version of the objective function
    """
    sdf_values, _ = vmap(lambda angles: calculate_arm_sdf(jnp.array([point]), angles, params_list))(angles_batch)
    return jnp.square(sdf_values).flatten()

@jit
def batch_objective_gradient(angles_batch, point, params_list):
    """
    Batch version of the gradient of the objective function
    """
    sdf_values, gradients, _ = vmap(lambda angles: calculate_arm_sdf_with_grad(jnp.array([point]), angles, params_list))(angles_batch)
    return 2 * sdf_values.reshape(-1, 1) * jnp.array(gradients)


def find_zero_sdf_angles(point, initial_angles_batch, params_list, num_attempts=2000, tolerance=1e-2):
    """
    Find multiple zero-level-set configurations for a given point using batch optimization.
    Returns both configurations and the index of the touching link.
    If no valid configurations are found, returns empty arrays.
    """
    bounds = [(-np.pi, np.pi)] + [(-np.pi/2, np.pi/2)] * (NUM_LINKS - 1)
    
    # Check if the number of initial angles is greater than num_attempts
    num_initial = len(initial_angles_batch)
    if num_initial > num_attempts:
        print(f"Warning: Number of initial angles ({num_initial}) is greater than num_attempts ({num_attempts}). "
              f"Only the first {num_attempts} initial angles will be used.")
        initial_angles_batch = initial_angles_batch[:num_attempts]
        num_initial = num_attempts

    # Generate initial guesses
    key = jax.random.PRNGKey(0)
    all_initial_angles = jax.random.uniform(
        key,
        shape=(num_attempts, NUM_LINKS),
        minval=jnp.array([-jnp.pi] + [-jnp.pi/2] * (NUM_LINKS - 1)),
        maxval=jnp.array([jnp.pi] + [jnp.pi/2] * (NUM_LINKS - 1))
    )
    
    # Set the provided initial angles
    for i, angles in enumerate(initial_angles_batch):
        all_initial_angles = all_initial_angles.at[i].set(jnp.array(angles))


    # Flatten the batch of initial angles
    flat_initial_angles = all_initial_angles.reshape(-1)

    # Define the objective function and gradient for the flattened batch
    def obj_func(flat_angles):
        angles_batch = flat_angles.reshape(num_attempts, NUM_LINKS)
        return jnp.sum(batch_objective_function(angles_batch, point, params_list))

    def obj_grad(flat_angles):
        angles_batch = flat_angles.reshape(num_attempts, NUM_LINKS)
        return batch_objective_gradient(angles_batch, point, params_list).reshape(-1)

    # Run optimization
    result = minimize(
        obj_func,
        flat_initial_angles,
        method='L-BFGS-B',
        jac=obj_grad,
        bounds=bounds * num_attempts,
        options={
            'ftol': 1e-6,
            'gtol': 1e-6,
            'maxiter': 1000,
            'maxfun': 15000,
        }
    )

    # Reshape the result back to a batch
    optimized_angles = result.x.reshape(num_attempts, NUM_LINKS)

    # Compute final SDF values
    final_sdf_values, touching_links = vmap(lambda angles: calculate_arm_sdf(jnp.array([point]), angles, params_list))(optimized_angles)


    # Filter results
    mask = jnp.abs(final_sdf_values) < tolerance
    zero_configs = optimized_angles[mask]
    touching_links = touching_links[mask]
    
    if len(zero_configs) == 0:
        # print(f"No zero-level set configurations found. Closest SDF: {jnp.min(jnp.abs(final_sdf_values)):.6f}")
        return jnp.array([]), jnp.array([])
    
    touching_links = touching_links + 1  # Add 1 because link indices are 1-based


    return zero_configs, touching_links



def compute_cdf_batch(q_batch, zero_config, touching_link):
    """
    Compute the CDF value as the Euclidean distance between partial configurations for a batch of q.
    """
    q_batch = np.atleast_2d(q_batch)
    zero_config = np.atleast_1d(zero_config)
    return np.linalg.norm(q_batch[:, :touching_link] - zero_config[:touching_link], axis=1)

def compute_cdf_gradient(q, zero_configs):
    """
    Compute the gradient of the CDF according to equation (6) in the paper.
    """
    min_distance = float('inf')
    q_min = None
    k_c = None
    
    for k in range(1, NUM_LINKS + 1):
        for q_prime in zero_configs:
            distance = np.linalg.norm(q[:k] - q_prime[:k])
            if distance < min_distance:
                min_distance = distance
                q_min = q_prime
                k_c = k
    
    if q_min is None:
        return np.zeros_like(q)
    
    gradient = np.zeros_like(q)
    gradient[:k_c] = (q[:k_c] - q_min[:k_c]) / np.linalg.norm(q[:k_c] - q_min[:k_c])
    
    return gradient

def find_closest_zero_config(point, initial_q_batch, params_list):
    """
    Find the closest zero-level-set configuration for a given point,
    using a batch of initial configurations or a single configuration.
    Returns batches of closest configurations, CDF values, and touching links.
    """
    initial_q_batch = np.atleast_2d(initial_q_batch)
    zero_configs, touching_links = find_zero_sdf_angles(point, initial_q_batch, params_list)
    
    if len(zero_configs) == 0:
        return None, None, None
    
    closest_configs = []
    cdf_values = []
    closest_touching_links = []
    
    for initial_q in initial_q_batch:
        initial_cdf_values = []
        for zero_config, touching_link in zip(zero_configs, touching_links):
            cdf_value = compute_cdf_batch(initial_q, zero_config, touching_link)[0]
            initial_cdf_values.append(cdf_value)
        
        min_index = np.argmin(initial_cdf_values)
        closest_configs.append(zero_configs[min_index])
        cdf_values.append(initial_cdf_values[min_index])
        closest_touching_links.append(touching_links[min_index])
    
    return np.array(closest_configs), np.array(cdf_values), np.array(closest_touching_links)

def visualize_arm_with_point(initial_angles, closest_angles, point, save_path='arm_with_point.png'):
    """
    Visualize the robot arm with initial and closest configurations, and the target point, and save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    for ax, angles, title in zip([ax1, ax2], [initial_angles, closest_angles], ['Initial Configuration', 'Closest Configuration']):
        joint_positions = forward_kinematics(angles)
        current_angle = 0
        
        for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
            if i < len(angles):
                current_angle += angles[i]
            transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
            ax.fill(*zip(*transformed_shape), alpha=0.5)
            ax.plot(joint_pos[0], joint_pos[1], 'bo', markersize=8)
        
        # Plot target point
        ax.plot(point[0], point[1], 'r*', markersize=15, label='Target Point')
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")

def visualize_arm_cdf(angles, params_list, save_path='arm_cdf_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Generate points for CDF evaluation
    # 5 links version: 
    # n_points = 100
    # xy_bound_box = 20
    # x = np.linspace(-xy_bound_box, xy_bound_box, n_points)
    # y = np.linspace(-xy_bound_box, xy_bound_box, n_points)

    # 2 links version: 
    n_points = 100
    xy_bound_box = 9
    x = np.linspace(-xy_bound_box, xy_bound_box, n_points)
    y = np.linspace(-xy_bound_box, xy_bound_box, n_points)

    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx.flatten(), yy.flatten()), axis=-1)
    
    # Calculate CDF values sequentially
    cdf_values = []
    start_time = time.time()
    last_print_time = start_time
    
    for i, point in enumerate(points):
        _, cdf_value, _ = find_closest_zero_config(point, angles, params_list)
        if cdf_value is None or len(cdf_value) == 0:
            cdf_value = float('inf')  # Use infinity for unreachable points
        else:
            cdf_value = float(cdf_value[0])  # Convert to scalar float
        
        if i % 100 == 0 and i > 0:
            current_time = time.time()
            elapsed_time = current_time - last_print_time
            print(f'step: {i}, cdf_value: {cdf_value}, time for last 100 iterations: {elapsed_time:.2f} seconds')
            last_print_time = current_time
        cdf_values.append(cdf_value)  # No need to check, as find_closest_zero_config already returns float('inf') for unreachable points
    
    total_time = time.time() - start_time
    print(f'Total time: {total_time:.2f} seconds')
    
    cdf_values = np.array(cdf_values).reshape(n_points, n_points)

    # Create heatmap
    heatmap = ax.imshow(cdf_values, cmap='viridis', extent=[-xy_bound_box, xy_bound_box, -xy_bound_box, xy_bound_box], origin='lower', aspect='equal', vmin=-0.5, vmax=5)
    
    # Plot robot arm
    joint_positions = forward_kinematics(angles)
    current_angle = 0
    
    for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
        if i < len(angles):
            current_angle += angles[i]
        transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
        ax.fill(*zip(*transformed_shape), alpha=0.5)
        ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title('CDF Visualization for Robot Arm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('CDF Value')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")

    
def main():
    # 5 links version: 
    # angles = np.array([np.pi/2, -np.pi/4, 0, np.pi/3 ,-np.pi/4])

    # 2 links version: 
    angles = np.array([np.pi/2, -np.pi/4])
    params_list = []
    for i in range(NUM_LINKS):
        params = np.load(f"trained_models/sdf_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)

    # Test points
    test_points = [
        np.array([7.0, 3.0]),
        np.array([-1.0, 4.0]),
        np.array([-4., -6.0]),
        np.array([4.0, -3.0]),
    ]

    # Batch of initial configurations
    initial_q_batch = np.array([
        [0, 0]
        # [np.pi/4, -np.pi/4, np.pi/4, -np.pi/4, 0],
        # [-np.pi/4, np.pi/4, -np.pi/4, np.pi/4, 0]
    ])

    for i, point in enumerate(test_points):
        print(f"\nFinding closest zero-level-set configurations for point: {point}")
        
        closest_configs, cdf_values, touching_links = find_closest_zero_config(point, initial_q_batch, params_list)
        
        if closest_configs is not None:
            print(f"Initial configurations:")
            for j, (initial_q, closest_config, cdf_value, touching_link) in enumerate(zip(initial_q_batch, closest_configs, cdf_values, touching_links)):
                print(f"  Initial q{j+1}: {initial_q}")
                print(f"  Closest config: {closest_config}")
                print(f"  CDF value: {cdf_value}")
                print(f"  Touching link: {touching_link}")
                print()
            
            # Visualize the arm with the target point for the best configuration
            best_index = np.argmin(cdf_values)
            visualize_arm_with_point(initial_q_batch[best_index], closest_configs[best_index], point, save_path=f'arm_with_point_{i+1}.png')
        else:
            print("No zero-level-set configurations found.")

    visualize_arm_cdf(angles, params_list)

if __name__ == "__main__":
    main()