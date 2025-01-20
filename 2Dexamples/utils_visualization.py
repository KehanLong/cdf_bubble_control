import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from typing import List
import torch

from utils_env import plot_environment


def visualize_results(obstacles, initial_config, goal_config, trajectory, src_dir):
    """Create visualizations of the planning results"""
    # Create workspace visualization
    fig_ws, ax_ws = plt.subplots(figsize=(12, 12))
    
    if trajectory is not None:
        # Plot trajectory in configuration space
        fig_cs, ax_cs = plt.subplots(figsize=(10, 10))
        trajectory = np.array(trajectory)
        
        # Plot trajectory in C-space
        ax_cs.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Path', alpha=0.7)
        ax_cs.plot(initial_config[0], initial_config[1], 'go', label='Start')
        ax_cs.plot(goal_config[0], goal_config[1], 'ro', label='Goal')
        
        ax_cs.set_xlabel('θ₁')
        ax_cs.set_ylabel('θ₂')
        ax_cs.set_title('Path in Configuration Space')
        ax_cs.grid(True)
        ax_cs.legend()
        
        # Save configuration space plot
        plt.savefig(os.path.join(src_dir, 'figures/config_space_path.png'))
        plt.close(fig_cs)
        
        # Print debug info
        # print("\nTrajectory debug info:")
        # print(f"Initial config: {initial_config}")
        # print(f"Goal config: {goal_config}")
        # print(f"First waypoint: {trajectory[0]}")
        # print(f"Last waypoint: {trajectory[-1]}")
        
        # Plot intermediate configurations in workspace
        num_viz_configs = min(10, len(trajectory))  # Show up to 10 intermediate configs
        viz_indices = np.linspace(0, len(trajectory)-1, num_viz_configs, dtype=int)
        
        for i, idx in enumerate(viz_indices):
            config = trajectory[idx]
            alpha = 0.3 if i not in [0, len(viz_indices)-1] else 0.7
            color = 'blue' if i not in [0, len(viz_indices)-1] else ('green' if i == 0 else 'red')
            label = 'Start' if i == 0 else ('Goal' if i == len(viz_indices)-1 else None)
            
            plot_environment(obstacles, config, ax=ax_ws, 
                           robot_color=color, 
                           robot_alpha=alpha,
                           plot_obstacles=(i==0),  # Only plot obstacles once
                           label=label)
    
    #ax_ws.set_title('Workspace Visualization')
    ax_ws.legend()
    
    # Save workspace visualization
    plt.savefig(os.path.join(src_dir, 'figures/workspace_visualization.png'))
    plt.close(fig_ws)
    
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

def visualize_cdf_bubble_planning(robot_cdf, initial_config, goal_configs, trajectory, bubbles, 
                          obstacle_points, src_dir, planner_type='bubble', resolution=50):
    """
    Visualize CDF planning in configuration space (theta1-theta2)
    """
    print("\nDebug - visualize_cdf_planning:")
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)
    
    # Reshape obstacle points to match expected format [B, N, 2]
    obstacle_points = torch.tensor(obstacle_points, device=robot_cdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    print(f"Obstacle points shape: {obstacle_points.shape}")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = resolution * resolution // batch_size
    cdf_values = np.zeros(resolution * resolution)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        # Create batch of configurations
        batch_configs = torch.tensor(
            [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
            device=robot_cdf.device,
            dtype=torch.float32
        ).unsqueeze(0)  # Shape: [1, batch_size, 2]
        
        # For each config in the batch, query CDF with all obstacle points
        for i in range(batch_configs.shape[1]):
            config = batch_configs[:, i:i+1, :]  # [1, 1, 2]
            cdf_val = robot_cdf.query_cdf(obstacle_points, config)
            cdf_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
    
    # Reshape back to grid
    cdf_values = cdf_values.reshape(resolution, resolution)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot CDF field without transpose (matches visualize_cdf_field)
    im = ax.imshow(cdf_values, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='CDF Value')
    
    # Plot bubbles from igraph Graph object
    if bubbles is not None:
        print(f"Plotting bubbles from graph with {len(bubbles.vs)} vertices")
        for vertex in bubbles.vs:
            circle = vertex["circle"]
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
    #ax.set_title('CDF Planning Visualization')
    ax.legend()

    if planner_type == 'bubble':
        plt.savefig(os.path.join(src_dir, 'figures/cdf_bubble_planning_visualization.png'), 
                    bbox_inches='tight', dpi=300)
    elif planner_type == 'bubble_connect':
        plt.savefig(os.path.join(src_dir, 'figures/connect_bubble_planning_visualization.png'), 
                    bbox_inches='tight', dpi=300)
    plt.close()

def visualize_cdf_field(robot_cdf, obstacle_points, resolution=100):
    """
    Visualize CDF field in configuration space (theta1-theta2)
    
    Args:
        robot_cdf: RobotCDF instance
        obstacle_points: numpy array of shape [N, 2] containing obstacle points
        resolution: int, number of points in each dimension
    """
    print("\nVisualizing CDF field...")
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)  # Swapped order to match visualization
    
    # Reshape obstacle points to match expected format [B, N, 2]
    obstacle_points = torch.tensor(obstacle_points, device=robot_cdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    print(f"Obstacle points shape: {obstacle_points.shape}")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = resolution * resolution // batch_size
    cdf_values = np.zeros(resolution * resolution)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        # Create batch of configurations
        batch_configs = torch.tensor(
            [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
            device=robot_cdf.device,
            dtype=torch.float32
        ).unsqueeze(0)  # Shape: [1, batch_size, 2]
        
        # For each config in the batch, query CDF with all obstacle points
        for i in range(batch_configs.shape[1]):
            config = batch_configs[:, i:i+1, :]  # [1, 1, 2]
            cdf_val = robot_cdf.query_cdf(obstacle_points, config)
            cdf_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
    
    # Reshape back to grid
    cdf_values = cdf_values.reshape(resolution, resolution)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot CDF field without transpose
    im = ax.imshow(cdf_values, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label='CDF Value')
    
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_title('CDF Field in Configuration Space')
    
    plt.show()
    
    return cdf_values

def visualize_sdf_field(robot_sdf, obstacle_points, resolution=100):
    """
    Visualize SDF field in configuration space (theta1-theta2)
    """
    print("\nVisualizing SDF field...")
    
    # Create a grid of configurations
    theta1 = torch.linspace(-np.pi, np.pi, resolution)
    theta2 = torch.linspace(-np.pi, np.pi, resolution)
    T1, T2 = torch.meshgrid(theta1, theta2, indexing='ij')
    
    # Reshape obstacle points to match expected format [1, N, 2]
    obstacle_points = torch.tensor(obstacle_points, device=robot_sdf.device, dtype=torch.float32)
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    
    # Process in batches to avoid memory issues
    batch_size = 500
    num_batches = resolution * resolution // batch_size
    sdf_values = torch.zeros(resolution * resolution, device=robot_sdf.device)
    
    for b in range(num_batches + 1):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, resolution * resolution)
        if start_idx >= end_idx:
            break
            
        # Create batch of configurations
        batch_configs = torch.stack([
            T1.flatten()[start_idx:end_idx],
            T2.flatten()[start_idx:end_idx]
        ], dim=1)  # Shape: [batch_size, 2]
        
        # For each config in batch, compute minimum SDF over all obstacle points
        batch_sdfs = []
        for i in range(batch_configs.shape[0]):
            config = batch_configs[i:i+1]  # Take one config [1, 2]
            sdf_val = robot_sdf.query_sdf(obstacle_points, config)  # [1, N]
            min_sdf = sdf_val.min()  # Minimum distance to any obstacle
            batch_sdfs.append(min_sdf)
        
        sdf_values[start_idx:end_idx] = torch.stack(batch_sdfs)
    
    # Reshape back to grid
    sdf_field = sdf_values.reshape(resolution, resolution).cpu()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot SDF field
    im = ax.imshow(sdf_field.T, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='RdBu', aspect='equal')
    plt.colorbar(im, ax=ax, label='SDF Value')
    
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_title('SDF Field in Configuration Space')

    plt.show()
    
        
    return sdf_field

def visualize_ompl_rrt_planning(robot_cdf, robot_sdf, initial_config, goal_configs, trajectory, 
                              obstacle_points, src_dir, planner_type='sdf_rrt', resolution=50):
    """
    Visualize OMPL RRT planning in configuration space (theta1-theta2)
    
    Args:
        robot_cdf: RobotCDF instance
        robot_sdf: RobotSDF instance
        initial_config: Initial configuration [2]
        goal_configs: List of goal configurations [N, 2]
        trajectory: Planned trajectory waypoints [M, 2]
        obstacle_points: Obstacle points tensor
        src_dir: Directory to save figures
        planner_type: 'cdf_rrt' or 'sdf_rrt'
        resolution: Grid resolution for field visualization
    """
    print(f"\nVisualizing {planner_type} planning...")
    
    # Create a grid of configurations
    theta1 = np.linspace(-np.pi, np.pi, resolution)
    theta2 = np.linspace(-np.pi, np.pi, resolution)
    T1, T2 = np.meshgrid(theta2, theta1)
    
    # Reshape obstacle points to [1, N, 2]
    obstacle_points = obstacle_points.reshape(1, -1, 2)
    
    # Get field values based on planner type
    if planner_type == 'cdf_rrt':
        field_values = np.zeros(resolution * resolution)
        batch_size = 100
        num_batches = resolution * resolution // batch_size
        
        for b in range(num_batches + 1):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, resolution * resolution)
            if start_idx >= end_idx:
                break
                
            # Create batch of configurations [batch_size, 2]
            batch_configs = torch.tensor(
                [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
                device=robot_cdf.device,
                dtype=torch.float32
            )
            
            for i in range(batch_configs.shape[0]):
                # Reshape config to [1, 1, 2] for CDF query
                config = batch_configs[i:i+1].unsqueeze(0)  # [1, 1, 2]
                cdf_val = robot_cdf.query_cdf(obstacle_points, config)
                field_values[start_idx + i] = cdf_val.min().detach().cpu().numpy()
        
        field_values = field_values.reshape(resolution, resolution)
        cmap = 'viridis'
        field_label = 'CDF Value'
        
    else:  # sdf_rrt
        field_values = np.zeros(resolution * resolution)
        batch_size = 100
        num_batches = resolution * resolution // batch_size
        
        for b in range(num_batches + 1):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, resolution * resolution)
            if start_idx >= end_idx:
                break
                
            # Create batch of configurations [batch_size, 2]
            batch_configs = torch.tensor(
                [[T1.flatten()[i], T2.flatten()[i]] for i in range(start_idx, end_idx)],
                device=robot_sdf.device,
                dtype=torch.float32
            )
            
            for i in range(batch_configs.shape[0]):
                # Reshape config to [1, 2] for SDF query
                config = batch_configs[i:i+1]  # [1, 2]
                sdf_val = robot_sdf.query_sdf(obstacle_points, config)
                field_values[start_idx + i] = sdf_val.min().detach().cpu().numpy()
        
        field_values = field_values.reshape(resolution, resolution)
        cmap = 'RdBu'
        field_label = 'SDF Value'
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot field
    im = ax.imshow(field_values, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap=cmap, aspect='equal')
    plt.colorbar(im, ax=ax, label=field_label)
    
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
    ax.legend()
    
    # Save figure
    if planner_type == 'cdf_rrt':
        plt.savefig(os.path.join(src_dir, 'figures/rrt_cdf_planning_visualization.png'), 
                    bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(src_dir, 'figures/rrt_sdf_planning_visualization.png'), 
                    bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Example usage
    from robot_cdf import RobotCDF
    from utils_env import create_obstacles

    from robot_sdf import RobotSDF

    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rng = np.random.default_rng(seed)
    
    # Create robot CDF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    # Create obstacles
    obstacles = create_obstacles(rng=rng)
    obstacle_points = np.concatenate(obstacles, axis=0)
    
    # Visualize CDF field
    cdf_values = visualize_cdf_field(robot_cdf, obstacle_points, resolution=100)
    
    # Visualize SDF field
    sdf_field = visualize_sdf_field(robot_sdf, obstacle_points, resolution=100)




