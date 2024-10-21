import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

class SimpleCDF2D:
    def __init__(self, link_lengths, num_precomputed=1000):
        self.link_lengths = np.array(link_lengths)
        self.num_joints = len(link_lengths)
        self.q_min = np.array([-np.pi] * self.num_joints)
        self.q_max = np.array([np.pi] * self.num_joints)
        self.num_precomputed = num_precomputed
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

    def precompute_configs(self, obstacle):
        #print("Precomputing configurations...")
        self.precomputed_configs = []
        attempts = 0
        max_attempts = self.num_precomputed * 20  # Limit total attempts

        while len(self.precomputed_configs) < self.num_precomputed and attempts < max_attempts:
            initial_q = np.random.uniform(self.q_min, self.q_max)
            zero_config = self.find_zero_sdf_angles(obstacle, initial_q)
            if zero_config is not None:
                # Check if this configuration is sufficiently different from existing ones
                if not self.precomputed_configs or min(np.linalg.norm(zero_config - existing, axis=0) for existing in self.precomputed_configs) > 0.02:
                    self.precomputed_configs.append(zero_config)
                    # if len(self.precomputed_configs) % 10 == 0:
                    #     print(f"Precomputed {len(self.precomputed_configs)}/{self.num_precomputed} configurations")
            attempts += 1

        self.precomputed_configs = np.array(self.precomputed_configs)
        #print(f"Precomputation complete. Found {len(self.precomputed_configs)} unique configurations.")

    def calculate_cdf(self, q, obstacle):
        if self.precomputed_configs is None:
            self.precompute_configs(obstacle)
        
        distances = np.linalg.norm(q - self.precomputed_configs, axis=1)
        min_distance = np.min(distances)
        
        # Determine the sign of the CDF
        sdf = self.calculate_sdf(q, obstacle)
        return min_distance if sdf >= 0 else -min_distance

def plot_robot_and_obstacle(ax, cdf, q, obstacle):
    # Plot robot arm
    x, y = cdf.forward_kinematics(q)
    ax.plot(x[0:2], y[0:2], 'b-', linewidth=2, label='Link 1')
    ax.plot(x[1:], y[1:], 'g-', linewidth=2, label='Link 2')
    ax.plot(x[0], y[0], 'ro', markersize=10, label='Base')
    ax.plot(x[1], y[1], 'bo', markersize=8, label='Joint')
    ax.plot(x[2], y[2], 'go', markersize=8, label='End effector')

    # Plot obstacle
    circle = plt.Circle(obstacle[:2], obstacle[2], color='r', fill=False, label='Obstacle')
    ax.add_artist(circle)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Arm and Obstacle')
    ax.legend()

def plot_field(ax, cdf, obstacle, field_type, resolution=50):
    print(f"Plotting {field_type}...")
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)

    Z = np.zeros_like(Theta1)
    total_points = resolution ** 2
    for idx, (i, j) in enumerate(np.ndindex(Theta1.shape)):
        if idx % 100 == 0:
            print(f"{field_type} computation: {idx}/{total_points} points processed")
        q = np.array([Theta1[i, j], Theta2[i, j]])
        if field_type == 'CDF':
            Z[i, j] = cdf.calculate_cdf(q, obstacle)
        else:  # SDF
            Z[i, j] = cdf.calculate_sdf(q, obstacle)

    contour = ax.contourf(Theta1, Theta2, Z, levels=20, cmap='viridis')
    zero_level = ax.contour(Theta1, Theta2, Z, levels=[0], colors='r', linewidths=2)
    plt.colorbar(contour, ax=ax, label='Distance')

    # Add labels to the zero level set
    ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    ax.set_xlabel('θ1')
    ax.set_ylabel('θ2')
    ax.set_title(f'{field_type}')

def main():
    print("Starting main function...")
    # Create a 2-link robot
    cdf = SimpleCDF2D([2, 2], num_precomputed=1000)

    # Define obstacle (circle: [center_x, center_y, radius])
    obstacle = [1.5, 1.5, 0.5]

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # Plot robot arm and obstacle
    q = np.array([0, np.pi/4])  # Example configuration
    plot_robot_and_obstacle(ax1, cdf, q, obstacle)

    # Plot SDF
    start_time = time.time()
    plot_field(ax2, cdf, obstacle, field_type='SDF')
    sdf_time = time.time() - start_time

    # Plot CDF
    start_time = time.time()
    plot_field(ax3, cdf, obstacle, field_type='CDF')
    cdf_time = time.time() - start_time

    plt.tight_layout()
    
    print(f"SDF computation time: {sdf_time:.2f} seconds")
    print(f"CDF computation time: {cdf_time:.2f} seconds")
    
    print("Saving plot...")
    plt.savefig('cdf_sdf_comparison.png')
    print("Showing plot...")
    plt.show()

if __name__ == "__main__":
    main()
