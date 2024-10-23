import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jaxopt import LBFGS

def plot_debug_configurations(ax, cdf, obstacle, num_samples=4):
    # Plot obstacle
    circle = plt.Circle(obstacle[:2], obstacle[2], color='r', fill=False, label='Obstacle')
    ax.add_artist(circle)

    # Plot a sample of precomputed configurations if available
    if cdf.precomputed_configs is not None and len(cdf.precomputed_configs) > 0:
        sample_indices = np.random.choice(len(cdf.precomputed_configs), min(num_samples, len(cdf.precomputed_configs)), replace=False)
        for q in cdf.precomputed_configs[sample_indices]:
            x, y = cdf.forward_kinematics(q)
            ax.plot(x, y, 'b-', alpha=0.2)
    else:
        print("No precomputed configurations available for plotting.")

    # Plot the robot's workspace boundary
    theta = np.linspace(0, 2*np.pi, 100)
    r = sum(cdf.link_lengths)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'k--', label='Workspace boundary')

    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Configurations and Obstacle')
    ax.legend()

class SimpleCDF2D:
    def __init__(self, link_lengths):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = len(link_lengths)
        self.q_min = np.array([-np.pi] * self.num_joints)
        self.q_max = np.array([np.pi] * self.num_joints)
        self.precomputed_configs = None

    def forward_kinematics(self, q):
        x = [0]
        y = [0]
        current_angle = 0
        for i in range(self.num_joints):
            current_angle += q[i]
            x.append(x[-1] + self.link_lengths[i] * np.cos(current_angle))
            y.append(y[-1] + self.link_lengths[i] * np.sin(current_angle))
        return np.array(x), np.array(y)

    def point_to_segment_distance(self, p, a, b):
        # Vector from a to b
        ab = b - a
        # Vector from a to p
        ap = p - a
        # Project ap onto ab
        proj = np.dot(ap, ab) / np.dot(ab, ab)
        # Clamp projection to [0, 1]
        proj = max(0, min(1, proj))
        # Compute the closest point on the segment
        closest = a + proj * ab
        # Return the distance from p to the closest point
        return np.linalg.norm(p - closest)

    def calculate_sdf(self, q, obstacle):
        x, y = self.forward_kinematics(q)
        obstacle_point = np.array(obstacle[:2])  # Only use x and y coordinates
        obstacle_radius = obstacle[2]
        
        min_distance = float('inf')
        for i in range(len(x) - 1):
            a = np.array([x[i], y[i]])
            b = np.array([x[i+1], y[i+1]])
            distance = self.point_to_segment_distance(obstacle_point, a, b)
            min_distance = min(min_distance, distance)
        
        return min_distance - obstacle_radius  # Subtract the obstacle radius

    def find_zero_sdf_angles(self, obstacle, initial_q, tolerance=1e-3):
        def objective(q):
            return self.calculate_sdf(q, obstacle) ** 2

        result = minimize(objective, initial_q, method='L-BFGS-B', bounds=list(zip(self.q_min, self.q_max)))
        if result.fun < tolerance:
            return result.x
        return None

    def calculate_cdf(self, q, obstacle):
        if self.precomputed_configs is None:
            raise ValueError("Precomputed configurations not set. Call set_precomputed_configs first.")
        
        distances = np.linalg.norm(q - self.precomputed_configs, axis=1)
        min_distance = np.min(distances)
        
        # Determine the sign of the CDF
        sdf = self.calculate_sdf(q, obstacle)
        #return min_distance if sdf >= 0 else -min_distance
        return min_distance 

    def set_precomputed_configs(self, configs):
        self.precomputed_configs = configs

# JAX-based functions for precomputation
@jit
def forward_kinematics_jax(q, link_lengths):
    x = jnp.zeros(len(link_lengths) + 1)
    y = jnp.zeros(len(link_lengths) + 1)
    current_angle = 0
    for i in range(len(link_lengths)):
        current_angle += q[i]
        x = x.at[i+1].set(x[i] + link_lengths[i] * jnp.cos(current_angle))
        y = y.at[i+1].set(y[i] + link_lengths[i] * jnp.sin(current_angle))
    return x, y

@jit
def point_to_segment_distance_jax(p, a, b):
    ab = b - a
    ap = p - a
    proj = jnp.dot(ap, ab) / jnp.dot(ab, ab)
    proj = jnp.clip(proj, 0, 1)
    closest = a + proj * ab
    return jnp.linalg.norm(p - closest)

@jit
def calculate_sdf_jax(q, obstacle, link_lengths):
    x, y = forward_kinematics_jax(q, link_lengths)
    obstacle_point = jnp.array(obstacle[:2])
    obstacle_radius = obstacle[2]
    
    distances = vmap(point_to_segment_distance_jax, (None, 0, 0))(
        obstacle_point,
        jnp.column_stack((x[:-1], y[:-1])),
        jnp.column_stack((x[1:], y[1:]))
    )
    
    min_distance = jnp.min(distances)
    return min_distance - obstacle_radius

@partial(jit, static_argnums=(1,))
def find_zero_sdf_angles_jax(initial_q, num_iterations, obstacle, link_lengths):
    def objective(q):
        return jnp.square(calculate_sdf_jax(q, obstacle, link_lengths))

    lbfgs = LBFGS(objective, maxiter=num_iterations)
    result = lbfgs.run(initial_q)
    return result.params

def precompute_configs_jax(cdf, obstacle, num_precomputed=1000, num_iterations=100, min_difference=0.02):
    key = jax.random.PRNGKey(0)
    
    @jit
    def generate_initial_q(key):
        return jax.random.uniform(key, (cdf.num_joints,), minval=-jnp.pi, maxval=jnp.pi)

    @jit
    def batch_find_zero_sdf(keys):
        initial_qs = vmap(generate_initial_q)(keys)
        return vmap(find_zero_sdf_angles_jax, (0, None, None, None))(initial_qs, num_iterations, obstacle, cdf.link_lengths)


    batch_size = 200
    configs = []
    attempts = 0
    max_attempts = num_precomputed * 10  # Limit total attempts

    while len(configs) < num_precomputed and attempts < max_attempts:
        key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, batch_size)
        batch_configs = batch_find_zero_sdf(batch_keys)
        
        # Filter configurations based on SDF
        sdfs = vmap(calculate_sdf_jax, (0, None, None))(batch_configs, obstacle, cdf.link_lengths)
        valid_configs = batch_configs[jnp.abs(sdfs) < 1e-3]
        
        # Convert to numpy for easier manipulation
        valid_configs_np = np.array(valid_configs)
        
        for config in valid_configs_np:
            if not configs or np.min(np.linalg.norm(np.array(configs) - config, axis=1)) > min_difference:
                configs.append(config)
                # if len(configs) % 100 == 0:
                #     print(f"Precomputed {len(configs)}/{num_precomputed} configurations")
                if len(configs) >= num_precomputed:
                    break
        
        attempts += batch_size
        
    configs = np.array(configs)
    print(f"Precomputation complete. Found {len(configs)} unique configurations in {attempts} attempts.")
    return jnp.array(configs[:num_precomputed])

def plot_robot_and_obstacle(ax, cdf, q, obstacle):
    # Plot robot arm
    x, y = cdf.forward_kinematics(q)
    colors = plt.cm.rainbow(np.linspace(0, 1, cdf.num_joints))
    
    for i in range(cdf.num_joints):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2, label=f'Link {i+1}')
    
    ax.plot(x[0], y[0], 'ko', markersize=10, label='Base')
    for i in range(1, cdf.num_joints):
        ax.plot(x[i], y[i], 'o', color=colors[i-1], markersize=8, label=f'Joint {i}')
    ax.plot(x[-1], y[-1], 'go', markersize=8, label='End effector')

    # Plot obstacle
    circle = plt.Circle(obstacle[:2], obstacle[2], color='r', fill=False, label='Obstacle')
    ax.add_artist(circle)

    ax.set_xlim(-sum(cdf.link_lengths), sum(cdf.link_lengths))
    ax.set_ylim(-sum(cdf.link_lengths), sum(cdf.link_lengths))
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Arm and Obstacle')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_field(ax, cdf, obstacle, field_type, joint_pair, fixed_angles, resolution=50):
    print(f"Plotting {field_type} for joints {joint_pair[0]} and {joint_pair[1]}...")
    
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)

    Z = np.zeros_like(Theta1)
    total_points = resolution ** 2
    for idx, (i, j) in enumerate(np.ndindex(Theta1.shape)):
        if idx % 100 == 0:
            print(f"{field_type} computation: {idx}/{total_points} points processed")
        
        q = np.array(fixed_angles)
        q[joint_pair[0]] = Theta1[i, j]
        q[joint_pair[1]] = Theta2[i, j]
        
        if field_type == 'CDF':
            Z[i, j] = cdf.calculate_cdf(q, obstacle)
        else:  # SDF
            Z[i, j] = cdf.calculate_sdf(q, obstacle)

    contour = ax.contourf(Theta1, Theta2, Z, levels=20, cmap='viridis')
    zero_level = ax.contour(Theta1, Theta2, Z, levels=[0.1], colors='r', linewidths=2)
    plt.colorbar(contour, ax=ax, label='Distance')

    # Add labels to the zero level set
    ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    ax.set_xlabel(f'θ{joint_pair[0]+1}')
    ax.set_ylabel(f'θ{joint_pair[1]+1}')
    ax.set_title(f'{field_type} (Joints {joint_pair[0]+1} and {joint_pair[1]+1})')

def main():
    print("Starting main function...")
    link_lengths = [2, 2]  # Example: 5-link robot
    cdf = SimpleCDF2D(link_lengths)

    obstacle = [1.5, 1.5, 1e-4]

    # Precompute configurations using JAX
    print("Precomputing configurations...")
    precomputed_configs = precompute_configs_jax(cdf, obstacle, num_precomputed=2000)
    cdf.set_precomputed_configs(np.array(precomputed_configs))

    # Create figure with 2x2 subplots for SDF
    fig_sdf, axs_sdf = plt.subplots(2, 2, figsize=(20, 20))
    axs_sdf = axs_sdf.ravel()

    # Create figure with 2x2 subplots for CDF
    fig_cdf, axs_cdf = plt.subplots(2, 2, figsize=(20, 20))
    axs_cdf = axs_cdf.ravel()

    # Define joint pairs to plot
    #joint_pairs = [(0, 1), (1, 2), (0, 2), (2, 3)]
    joint_pairs = [(0, 1), (0, 1), (0, 1), (0, 1)]

    # Plot SDF and CDF for each joint pair
    fixed_angles = np.zeros(cdf.num_joints)
    for i, joint_pair in enumerate(joint_pairs):
        # Plot SDF
        plot_field(axs_sdf[i], cdf, obstacle, field_type='SDF', joint_pair=joint_pair, fixed_angles=fixed_angles)
        
        # Plot CDF
        plot_field(axs_cdf[i], cdf, obstacle, field_type='CDF', joint_pair=joint_pair, fixed_angles=fixed_angles)

    # Adjust layout and save SDF figure
    fig_sdf.tight_layout()
    print("Saving SDF plot...")
    fig_sdf.savefig('multi_joint_sdf_comparison.png')

    # Adjust layout and save CDF figure
    fig_cdf.tight_layout()
    print("Saving CDF plot...")
    fig_cdf.savefig('multi_joint_cdf_comparison.png')

    # Create debug figure
    # fig_debug, ax_debug = plt.subplots(figsize=(10, 10))
    # plot_debug_configurations(ax_debug, cdf, obstacle)
    # plt.savefig('debug_configurations.png')

    print("Showing plots...")
    plt.show()

if __name__ == "__main__":
    main()
