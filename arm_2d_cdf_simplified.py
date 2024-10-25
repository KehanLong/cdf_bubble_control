import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jaxopt import LBFGS

from utils_new import forward_kinematics_jax, point_to_segment_distance_jax


class SimpleCDF2D:
    def __init__(self, link_lengths):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = len(link_lengths)
        self.q_min = np.array([-np.pi] * self.num_joints)
        self.q_max = np.array([np.pi] * self.num_joints)
        self.precomputed_configs = None
        self.precomputed_touching_indices = None

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

    def calculate_cdf(self, q):
        if self.precomputed_configs is None or self.precomputed_touching_indices is None:
            raise ValueError("Precomputed configurations not set. Call set_precomputed_configs first.")
        
        # Vectorized partial distance calculation
        q_expanded = np.expand_dims(q, axis=0)  # Shape: (1, num_joints)
        configs_expanded = self.precomputed_configs  # Shape: (num_configs, num_joints)
        touch_indices_expanded = np.expand_dims(self.precomputed_touching_indices, axis=1)  # Shape: (num_configs, 1)
        
        # Create a mask for valid joints
        joint_indices = np.arange(self.num_joints)
        mask = joint_indices <= touch_indices_expanded
        
        # Calculate squared differences
        squared_diff = np.square(q_expanded - configs_expanded)
        
        # Apply mask and sum
        masked_squared_diff = squared_diff * mask
        partial_distances = np.sqrt(np.sum(masked_squared_diff, axis=1))
        
        # Find the minimum distance
        return np.min(partial_distances)

    def set_precomputed_configs(self, configs, touching_indices):
        self.precomputed_configs = configs
        self.precomputed_touching_indices = touching_indices

# JAX-based functions for precomputation

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
    
    # Calculate distances for each link
    x, y = forward_kinematics_jax(result.params, link_lengths)
    obstacle_point = jnp.array(obstacle[:2])
    distances = vmap(point_to_segment_distance_jax, (None, 0, 0))(
        obstacle_point,
        jnp.column_stack((x[:-1], y[:-1])),
        jnp.column_stack((x[1:], y[1:]))
    )
    
    # Find the index of the link with the minimum distance
    touching_link_index = jnp.argmin(distances)
    
    return result.params, touching_link_index

def precompute_configs_jax(cdf, obstacle, num_precomputed=1000, num_iterations=1000, min_difference=0.05):
    key = jax.random.PRNGKey(0)
    
    @jit
    def generate_initial_q(key):
        return jax.random.uniform(key, (cdf.num_joints,), minval=-jnp.pi, maxval=jnp.pi)

    @jit
    def batch_find_zero_sdf(keys):
        initial_qs = vmap(generate_initial_q)(keys)
        return vmap(find_zero_sdf_angles_jax, (0, None, None, None))(initial_qs, num_iterations, obstacle, cdf.link_lengths)

    batch_size = 1000
    configs = []
    touching_indices = []
    attempts = 0
    max_attempts = num_precomputed * 10  # Limit total attempts

    while len(configs) < num_precomputed and attempts < max_attempts:
        key, subkey = jax.random.split(key)
        batch_keys = jax.random.split(subkey, batch_size)
        batch_configs, batch_indices = batch_find_zero_sdf(batch_keys)
        
        # Filter configurations based on SDF
        sdfs = vmap(calculate_sdf_jax, (0, None, None))(batch_configs, obstacle, cdf.link_lengths)
        valid_mask = jnp.abs(sdfs) < 1e-3
        valid_configs = batch_configs[valid_mask]
        valid_indices = batch_indices[valid_mask]
        
        # Convert to numpy for easier manipulation
        valid_configs_np = np.array(valid_configs)
        valid_indices_np = np.array(valid_indices)
        
        for config, index in zip(valid_configs_np, valid_indices_np):
            if not configs or np.min(np.linalg.norm(np.array(configs) - config, axis=1)) > min_difference:
                configs.append(config)
                touching_indices.append(index)
                if len(configs) >= num_precomputed:
                    break
        
        attempts += batch_size
        
    configs = np.array(configs)
    touching_indices = np.array(touching_indices)
    print(f"Precomputation complete. Found {len(configs)} unique configurations in {attempts} attempts.")
    return jnp.array(configs[:num_precomputed]), jnp.array(touching_indices[:num_precomputed])

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
            Z[i, j] = cdf.calculate_cdf(q)
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


def plot_robot_configs(ax, cdf, configs, obstacle):
    ax.clear()
    circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], fill=False)
    ax.add_artist(circle)
    ax.set_xlim(-sum(cdf.link_lengths), sum(cdf.link_lengths))
    ax.set_ylim(-sum(cdf.link_lengths), sum(cdf.link_lengths))
    ax.set_aspect('equal')
    
    for config in configs:
        x, y = cdf.forward_kinematics(config)
        ax.plot(x, y, 'b-', alpha=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def main():
    print("Starting main function...")
    link_lengths = [2., 2., 2., 2.]  # 4-link robot
    cdf = SimpleCDF2D(link_lengths)

    obstacle = [2.0, 2.0, 0.5]  # [x, y, radius]

    # Precompute configurations
    configs, touching_indices = precompute_configs_jax(cdf, obstacle, num_precomputed=4000)
    cdf.set_precomputed_configs(np.array(configs), np.array(touching_indices))

    # Create figure with 3x2 subplots for SDF
    fig_sdf, axs_sdf = plt.subplots(3, 2, figsize=(20, 30))
    axs_sdf = axs_sdf.ravel()

    # Create figure with 3x2 subplots for CDF
    fig_cdf, axs_cdf = plt.subplots(3, 2, figsize=(20, 30))
    axs_cdf = axs_cdf.ravel()

    # Define joint pairs to plot
    joint_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Plot SDF and CDF for each joint pair
    fixed_angles = np.zeros(cdf.num_joints)
    for i, joint_pair in enumerate(joint_pairs):
        # Set fixed angles for other joints
        fixed_angles = np.zeros(cdf.num_joints)
        fixed_joints = [j for j in range(cdf.num_joints) if j not in joint_pair]
        fixed_angles[fixed_joints] = np.pi / 4  # Set fixed joints to 45 degrees

        # Plot SDF
        plot_field(axs_sdf[i], cdf, obstacle, field_type='SDF', joint_pair=joint_pair, fixed_angles=fixed_angles)
        
        # Plot CDF
        plot_field(axs_cdf[i], cdf, obstacle, field_type='CDF', joint_pair=joint_pair, fixed_angles=fixed_angles)

    # Adjust layout and save SDF figure
    fig_sdf.tight_layout()
    print("Saving SDF plot...")
    fig_sdf.savefig('multi_joint_sdf_comparison.png', dpi=300, bbox_inches='tight')

    # Adjust layout and save CDF figure
    fig_cdf.tight_layout()
    print("Saving CDF plot...")
    fig_cdf.savefig('multi_joint_cdf_comparison.png', dpi=300, bbox_inches='tight')

    print("Analysis complete. Check 'multi_joint_sdf_comparison.png' and 'multi_joint_cdf_comparison.png' for results.")


    # fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # # Plot robot configurations in workspace
    # ax_workspace = axs[0]
    # plot_robot_configs(ax_workspace, cdf, configs, obstacle)
    # ax_workspace.set_title('Robot Configurations')
    
    # # Plot CDF
    # ax_cdf = axs[1]
    # plot_field(ax_cdf, cdf, obstacle, field_type='CDF', joint_pair=(0, 1), fixed_angles=np.zeros(cdf.num_joints))
    # ax_cdf.set_title('CDF')

    # fig.tight_layout()
    # plt.savefig('original_sampling_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close(fig)


if __name__ == "__main__":
    main()

