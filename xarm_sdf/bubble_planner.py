import pybullet as p
import numpy as np
import torch
import cvxpy
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cProfile
from sdf_marching.samplers import get_rapidly_exploring, get_uniform_random, get_rapidly_exploring_connect
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.samplers.tracing import trace_toward_graph_all
from sdf_marching.discrete import get_shortest_path
from sdf_marching.cvx import edgeseq_to_traj_constraint_bezier, bezier_cost_all

@dataclass
class Bubble:
    """Represents a configuration space bubble"""
    center: np.ndarray  # Configuration at bubble center
    radius: float       # Bubble radius in configuration space

class BubblePlanner:
    def __init__(self, robot_cdf, joint_limits, device='cuda', random_seed=42):
        """Initialize the bubble planner using CDF for collision checking"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Store robot info
        self.robot_cdf = robot_cdf
        self.joint_limits = joint_limits
        self.device = device
        
        # Planning parameters
        self.epsilon = 5E-2          # Bubble expansion parameter
        self.min_radius = 1E-2       # Minimum bubble radius
        self.max_cdf = 1.0        # Maximum trusted CDF radius
        self.num_samples = 5E4
        self.max_iterations = 5E4
        self.step_size = 0.2
        self.goal_bias = 0.1

    def query_cdf(self, config: np.ndarray, obstacle_points: torch.Tensor) -> float:
        """Query CDF value for a given configuration"""
        # Convert config to tensor and ensure [B, 6] shape
        config_tensor = torch.tensor(config, device=self.device, dtype=torch.float32)
        if config_tensor.dim() == 1:
            config_tensor = config_tensor.unsqueeze(0)  # [1, 6]
        
        # Ensure points are [B, N, 3]
        if obstacle_points.dim() == 2:
            obstacle_points = obstacle_points.unsqueeze(0)  # [1, N, 3]
        
        
        # Query CDF model
        cdf_values = self.robot_cdf.query_sdf(
            points=obstacle_points,      # [B, N, 3]
            joint_angles=config_tensor,  # [B, 6]
            return_gradients=False
        )
        
        min_cdf = cdf_values.min().detach().cpu().numpy()

        min_cdf = min(min_cdf * 5, self.max_cdf)

        #print(f"Min CDF: {min_cdf}")
        
        return min_cdf

    def get_bubble(self, config: np.ndarray, points: torch.Tensor) -> Bubble:
        """Generate a bubble at the given configuration using CDF with radius capping"""
        # Query CDF to get distance to obstacles
        distance = self.query_cdf(config, points)
        
        # Cap the radius to ensure we only use reliable CDF values
        radius = max(distance - self.epsilon, self.min_radius)
        
        return Bubble(center=config, radius=radius)

    def bubbles_connect(self, b1: Bubble, b2: Bubble) -> bool:
        """Check if two bubbles can be connected"""
        # Check if bubbles overlap
        center_dist = np.linalg.norm(b1.center - b2.center)
        return center_dist <= (b1.radius + b2.radius + self.connection_threshold)

    def interpolate_path(self, b1: Bubble, b2: Bubble, num_points: int = 10) -> np.ndarray:
        """Generate interpolated path between two connected bubbles"""
        return np.linspace(b1.center, b2.center, num_points)

    def generate_bubbles(self, start_config: np.ndarray, goal_config: np.ndarray, obstacle_points: torch.Tensor):
        """Generate bubbles using RRT-Connect"""
        # Convert configs to float32
        start_config = start_config.astype(np.float32)[:6]
        goal_config = goal_config.astype(np.float32)[:6]
        
        # Wrap CDF query for compatibility
        def cdf_wrapper(x):
            if len(x.shape) > 1:
                return np.array([self.query_cdf(xi, obstacle_points) for xi in x])
            return self.query_cdf(x, obstacle_points)
        
        try:
            # Create RNG with seed
            rng = np.random.default_rng(self.random_seed)
            
            # Generate bubbles using RRT-Connect
            overlaps_graph, max_circles, _ = get_rapidly_exploring_connect(
                cdf_wrapper,
                self.epsilon,
                self.min_radius,
                int(self.num_samples),
                self.joint_limits[0],
                self.joint_limits[1],
                start_point=start_config,
                batch_size=100,
                max_retry=500,
                max_retry_epsilon=100,
                max_num_iterations=int(self.max_iterations),
                inflate_factor=1.0,
                prc=0.1,
                end_point=goal_config,
                rng=rng
            )
            
            print(f"\nBubble generation complete:")
            print(f"Number of bubbles: {len(overlaps_graph.vs)}")
            return overlaps_graph, max_circles
            
        except Exception as e:
            print(f"Bubble generation failed: {str(e)}")
            raise e

    def find_path(self, bubbles: List[Bubble]) -> np.ndarray:
        """Find path through connected bubbles using simple waypoint interpolation"""
        path = []
        for i in range(len(bubbles) - 1):
            segment = self.interpolate_path(bubbles[i], bubbles[i+1])
            path.extend(segment)
        return np.array(path)

    def plan(self, start_config: np.ndarray, goal_config: np.ndarray, obstacle_points: torch.Tensor):
        """Plan a path from start to goal configuration"""
        print("\nStarting bubble-based planning...")
        
        try:
            # Generate bubbles with obstacle points
            overlaps_graph, max_circles = self.generate_bubbles(start_config, goal_config, obstacle_points)
            
            # Find start and goal indices
            start_idx = position_to_max_circle_idx(overlaps_graph, start_config)
            end_idx = position_to_max_circle_idx(overlaps_graph, goal_config)
            
            # Connect start/goal if needed
            if start_idx < 0:
                overlaps_graph, start_idx = trace_toward_graph_all(
                    overlaps_graph, 
                    lambda x: self.query_cdf(x, obstacle_points),  # Use query_cdf with points
                    self.epsilon, 
                    self.min_radius, 
                    start_config
                )
            
            if end_idx < 0:
                centers = np.array([v['circle'].centre for v in overlaps_graph.vs])
                end_idx = np.argmin(np.linalg.norm(centers - goal_config, axis=1))
                goal_config = centers[end_idx].copy()
            
            # Find shortest path
            overlaps_graph.to_directed()
            path_result = get_shortest_path(
                lambda from_circle, to_circle: from_circle.hausdorff_distance_to(to_circle),
                overlaps_graph,
                start_idx,
                end_idx,
                cost_name="cost",
                return_epath=True,
            )
            
            # Optimize trajectory
            bps, constr_bps = edgeseq_to_traj_constraint_bezier(
                overlaps_graph.es[path_result[0]], 
                start_config, 
                goal_config
            )
            
            cost = bezier_cost_all(bps)
            prob = cvxpy.Problem(cvxpy.Minimize(cost), constr_bps)
            prob.solve()
            
            # Generate final trajectory
            times = np.linspace(0, 1.0, 50)
            trajectory = np.vstack([bp.query(times).value for bp in bps])
            
            print(f"Planning complete! Generated trajectory with {len(trajectory)} waypoints")
            return trajectory
            
        except Exception as e:
            print(f"Planning failed: {str(e)}")
            return None

def main():
    """Test function"""
    print("Bubble planner module. Import and use with XArm visualizer.")

if __name__ == "__main__":
    main() 