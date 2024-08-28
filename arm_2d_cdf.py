import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.arm_2d_config import NUM_LINKS, shapes

from scipy.optimize import minimize
import jax.numpy as jnp
import jax

# Import necessary functions from arm_2d_utils.py
from arm_2d_utils import (
    calculate_arm_sdf,
    calculate_arm_sdf_with_grad,
    forward_kinematics,
    transform_shape
)


def objective_function(angles, point, params_list):
    """
    Objective function to minimize: squared SDF value
    """
    sdf_value = calculate_arm_sdf(jnp.array([point]), jnp.array(angles), params_list)

    return float(sdf_value ** 2)  # Convert to float for scipy.optimize

def objective_gradient(angles, point, params_list):
    """
    Gradient of the objective function
    """
    sdf_value, gradient = calculate_arm_sdf_with_grad(jnp.array([point]), jnp.array(angles), params_list)
    
    # squared version 
    return 2 * float(sdf_value) * np.array(gradient)  # Convert to numpy array for scipy.optimize



def find_zero_sdf_angles(point, initial_angles, params_list, num_attempts=10, tolerance=1e-6):
    """
    Find multiple zero-level-set configurations for a given point.
    """
    bounds = [(-np.pi, np.pi)] + [(-np.pi/2, np.pi/2)] * (NUM_LINKS - 1)
    zero_configs = []
    
    for attempt in range(num_attempts):
        if attempt == 0:
            angles = initial_angles
        else:
            # Generate a new random initial guess
            angles = np.random.uniform(-np.pi, np.pi, 1).tolist() + \
                     np.random.uniform(-np.pi/2, np.pi/2, NUM_LINKS-1).tolist()
        
        result = minimize(
            objective_function,
            angles,
            args=(point, params_list),
            method='L-BFGS-B',
            jac=objective_gradient,
            bounds=bounds,
            options={
                'ftol': 1e-8,
                'gtol': 1e-8,
                'maxiter': 1000,
                'maxfun': 15000,
            }
        )
        
        if result.fun < tolerance:  # Consider it a zero-level-set if absolute SDF is close to zero
            zero_configs.append(result.x)
    
    return np.array(zero_configs)

def compute_cdf(q, zero_configs):
    """
    Compute the CDF value according to equation (5) in the paper.
    """
    min_distance = float('inf')
    for k in range(1, NUM_LINKS + 1):
        for q_prime in zero_configs:
            distance = np.linalg.norm(q[:k] - q_prime[:k])
            if distance < min_distance:
                min_distance = distance
                # If we've found a zero configuration that matches up to this link,
                # we can stop checking further links
                if distance == 0:
                    return 0
    return min_distance

def find_closest_zero_config(point, initial_q, params_list):
    """
    Find zero-level-set configurations and compute the CDF value.
    """
    zero_configs = find_zero_sdf_angles(point, initial_q, params_list)
    
    if len(zero_configs) == 0:
        return None, None
    
    cdf_value = compute_cdf(initial_q, zero_configs)
    
    # Find the closest configuration (for visualization purposes)
    closest_config = min(zero_configs, key=lambda q: np.linalg.norm(initial_q - q))
    
    return closest_config, cdf_value



def visualize_arm_with_point(angles, point, save_path='arm_with_point.png'):
    """
    Visualize the robot arm with given angles and the target point, and save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
    ax.set_title('Robot Arm Configuration')
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
    n_points = 30
    x = np.linspace(-20, 20, n_points)
    y = np.linspace(-20, 20, n_points)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx.flatten(), yy.flatten()), axis=-1)
    
    # Calculate CDF values sequentially
    cdf_values = []
    for point in points:
        _, cdf_value = find_closest_zero_config(point, angles, params_list)
        print(cdf_value)
        cdf_values.append(cdf_value if cdf_value is not None else np.inf)
    
    cdf_values = np.array(cdf_values).reshape(n_points, n_points)

    # Create heatmap
    heatmap = ax.imshow(cdf_values, cmap='viridis', extent=[-20, 20, -20, 20], origin='lower', aspect='equal', vmin=-2, vmax=2)
    
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
    # Load parameters
    angles = jnp.array([np.pi/2, -np.pi/4, 0, np.pi/3 ,-np.pi/4])
    params_list = []
    for i in range(NUM_LINKS):
        params = jnp.load(f"trained_models/link{i+1}_model_4_16.npy", allow_pickle=True).item()
        params_list.append(params)

    # Test points
    test_points = [
        jnp.array([5.0, 7.0]),
        jnp.array([-1.0, 4.0]),
        jnp.array([-4., -12.0]),
        jnp.array([8.0, -10.0]),
    ]

    for i, point in enumerate(test_points):
        print(f"\nFinding closest zero-level-set configuration for point: {point}")
        
        initial_q = jnp.array([0, 0, 0, 0, 0])
        closest_config, cdf_value = find_closest_zero_config(point, initial_q, params_list)
        
        if closest_config is not None:
            print(f"Initial configuration: {initial_q}")
            print(f"Closest zero-level-set configuration: {closest_config}")
            print(f"CDF value: {cdf_value}")
            
            # Visualize the arm with the target point and save the figure
            visualize_arm_with_point(closest_config, point, save_path=f'arm_with_point_{i+1}.png')
        else:
            print("No zero-level-set configuration found.")

    visualize_arm_cdf(angles, params_list)

if __name__ == "__main__":
    main()