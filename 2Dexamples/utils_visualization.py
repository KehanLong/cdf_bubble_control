import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from typing import List
import torch

from utils_env import plot_environment


def visualize_results(obstacles, initial_config, goal_config, trajectory, bezier_curves, src_dir):
    """Create visualizations of the planning results"""
    # Create workspace visualization
    fig_ws, ax_ws = plt.subplots(figsize=(12, 12))
    
    if trajectory is not None:
        # Plot intermediate configurations
        num_samples = 8
        indices = np.linspace(0, len(trajectory)-1, num_samples, dtype=int)
        
        for i, idx in enumerate(indices[1:-1]):
            config = trajectory[idx]
            alpha = 0.9 - ((i) / (num_samples-3)) * 0.6
            plot_environment(obstacles, config, ax=ax_ws, 
                           robot_color='gray', 
                           robot_alpha=alpha,
                           plot_obstacles=(i==0),
                           add_to_legend=False)
        
        # Plot initial trajectory point
        plot_environment(obstacles, trajectory[indices[0]], ax=ax_ws,
                        robot_color='gray',
                        robot_alpha=1.0,
                        plot_obstacles=False,
                        label='Planned Path')
    
    # Plot initial and goal configurations
    plot_environment(obstacles, initial_config, ax=ax_ws, 
                    goal_angles=goal_config,
                    robot_color='blue', 
                    robot_alpha=1.0,
                    plot_obstacles=False,
                    label='Initial Config',
                    save_path=os.path.join(src_dir, 'figures/workspace_visualization.png'))
    
    print("Saved workspace visualization")

def plot_path_comparison(planned_configs, tracked_configs, src_dir):
    """
    Create a visualization comparing planned and executed paths.
    
    Args:
        planned_configs: Array of planned configurations (N, 2)
        tracked_configs: Array of tracked/executed configurations (M, 2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot in configuration space (left plot)
    ax1.plot(planned_configs[:, 0], planned_configs[:, 1], 
             'b-', linewidth=2, label='Planned Path')
    ax1.plot(tracked_configs[:, 0], tracked_configs[:, 1], 
             'r--', linewidth=2, label='Executed Path')
    
    # Add start and end points
    ax1.plot(planned_configs[0, 0], planned_configs[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(planned_configs[-1, 0], planned_configs[-1, 1], 'ro', markersize=10, label='Goal')
    
    ax1.set_xlabel('θ₁', fontsize=14)
    ax1.set_ylabel('θ₂', fontsize=14)
    ax1.set_title('Configuration Space Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    
    # Plot tracking error over time (right plot)
    # Interpolate planned path to match length of tracked path
    from scipy.interpolate import interp1d
    planned_times = np.linspace(0, 1, len(planned_configs))
    tracked_times = np.linspace(0, 1, len(tracked_configs))
    
    # Create interpolation function for each joint
    interp_funcs = [interp1d(planned_times, planned_configs[:, i]) 
                   for i in range(planned_configs.shape[1])]
    
    # Get interpolated planned configurations at tracked times
    interpolated_planned = np.column_stack([f(tracked_times) for f in interp_funcs])
    
    # Compute tracking error
    tracking_error = np.linalg.norm(tracked_configs - interpolated_planned, axis=1)
    
    ax2.plot(tracked_times, tracking_error, 'k-', linewidth=2)
    ax2.set_xlabel('Normalized Time', fontsize=14)
    ax2.set_ylabel('Tracking Error', fontsize=14)
    ax2.set_title('Tracking Error Over Time', fontsize=16)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(src_dir, 'figures/path_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved path comparison plot")
    plt.close()

def create_animation(obstacles: List[np.ndarray], tracked_configs, reference_configs, 
                    dt: float = 0.02, src_dir=None):
    """
    Create an animation of the robot arm tracking the planned path.
    
    Args:
        obstacles: List of obstacle arrays
        tracked_configs: Array of tracked configurations
        reference_configs: Array of reference configurations
        dt: Time step between frames (seconds)
        src_dir: Source directory for saving the animation
    """
    # Calculate fps based on the timestep
    fps = int(1/dt)
    
    n_configs = len(tracked_configs)
    print(f"\nCreating animation for {n_configs} configurations (dt={dt}s, fps={fps})")
    
    # Create frames based on the number of configurations
    frames = []
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(n_configs):
        ax.clear()
        current_config = tracked_configs[i]
        
        # Plot the environment and current configuration
        plot_environment(obstacles, current_config, ax=ax, robot_color='blue', label='Current')
        
        # Plot reference configuration
        plot_environment(obstacles, reference_configs[i], ax=ax, robot_color='green', 
                        plot_obstacles=False, label='Reference', robot_alpha=0.5)
        
        # Set consistent axis limits
        ax.set_title(f'Frame {i}/{n_configs}')
        ax.legend()
        
        # Convert plot to RGB array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    
    if src_dir:
        # Save animation
        print("Saving animation...")
        output_path = os.path.join(src_dir, 'figures/robot_animation.mp4')
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Animation saved as '{output_path}'")
    
    return frames

def visualize_cdf_planning(robot_cdf, initial_config, goal_configs, trajectory, bubbles, 
                          obstacle_points, src_dir, resolution=100):
    """
    Visualize CDF planning in configuration space (theta1-theta2)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # Evaluate CDF values for each configuration
    cdf_values = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            config = torch.tensor(
                [[T1[j, i], T2[j, i]]],
                device=robot_cdf.device,
                dtype=torch.float32
            )
            cdf_values[i, j] = robot_cdf.query_cdf(
                obstacle_points.unsqueeze(0).unsqueeze(0),
                config
            ).cpu().numpy().min()
    
    # Plot CDF field with transposed values
    im = ax.imshow(cdf_values.T, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='CDF Value')
    
    # Plot bubbles from igraph Graph object
    if bubbles is not None:  # bubbles is actually an igraph Graph
        print(f"Plotting bubbles from graph with {len(bubbles.vs)} vertices")
        for vertex in bubbles.vs:
            circle = vertex["circle"]  # Each vertex has a circle attribute
            if circle is not None:
                center = circle.centre
                radius = circle.radius
                circle_patch = plt.Circle(
                    (center[0], center[1]), 
                    radius, 
                    fill=False, 
                    color='cyan', 
                    alpha=0.5
                )
                ax.add_patch(circle_patch)
    
    # Plot trajectory if available
    if trajectory is not None:
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Planned Path')
    
    # Plot start and goals
    ax.plot(initial_config[0], initial_config[1], 'go', markersize=10, label='Start')
    for i, goal in enumerate(goal_configs):
        ax.plot(goal[0], goal[1], 'r^', markersize=10, label=f'Goal {i+1}')
    
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_title('CDF Planning Visualization')
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(src_dir, 'figures/cdf_planning_visualization.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()




