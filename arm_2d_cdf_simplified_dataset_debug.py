import numpy as np
import matplotlib.pyplot as plt
from arm_2d_cdf_simplified import SimpleCDF2D

class CDFDatasetGenerator:
    def __init__(self, link_lengths, num_zero_configs):
        self.cdf = SimpleCDF2D(link_lengths, num_precomputed=num_zero_configs)
        self.workspace_radius = np.sum(link_lengths)

    def generate_workspace_point(self):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, self.workspace_radius)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y])

    def generate_config(self):
        return np.random.uniform(-np.pi, np.pi, self.cdf.num_joints)

    def compute_zero_configs(self, point):
        obstacle = [point[0], point[1], 0]
        self.cdf.precompute_configs(obstacle)
        return self.cdf.precomputed_configs

def visualize_debug(point, config, zero_configs, cdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the point
    ax.plot(point[0], point[1], 'ro', markersize=10, label='Target Point')
    
    # Plot zero configurations
    for i, zc in enumerate(zero_configs):
        x, y = cdf.forward_kinematics(zc)
        ax.plot(x, y, 'b-', alpha=0.5)
        ax.plot(x[-1], y[-1], 'bx', markersize=8)
        if i == 0:
            ax.plot([], [], 'b-', label='Zero Configs')
    
    # Plot the random configuration
    x, y = cdf.forward_kinematics(config)
    ax.plot(x, y, 'g-', linewidth=2, label='Random Config')
    ax.plot(x[-1], y[-1], 'go', markersize=8)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Debug Visualization: Point, Zero Configs, and Random Config')
    ax.grid(True)
    
    plt.show()

def main():
    link_lengths = [2, 2]  # 2-link robot arm
    num_zero_configs = 30

    generator = CDFDatasetGenerator(link_lengths, num_zero_configs)
    
    # Generate a single point
    point = generator.generate_workspace_point()
    print(f"Generated point: {point}")

    # Generate a single random configuration
    config = generator.generate_config()
    print(f"Generated random configuration: {config}")

    # Compute zero configurations
    zero_configs = generator.compute_zero_configs(point)
    print(f"Number of zero configurations: {len(zero_configs)}")
    print(zero_configs)

    # Visualize
    visualize_debug(point, config, zero_configs, generator.cdf)

if __name__ == "__main__":
    main()
