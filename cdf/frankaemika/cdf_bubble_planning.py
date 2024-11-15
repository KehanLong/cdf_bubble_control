import pybullet as p
import numpy as np
import torch
import time
from typing import List, Tuple
import cvxpy
from dataclasses import dataclass
from sdf_marching.samplers import get_rapidly_exploring
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.samplers.tracing import trace_toward_graph_all
from sdf_marching.discrete import get_shortest_path
from sdf_marching.cvx import edgeseq_to_traj_constraint_bezier, bezier_cost_all

# Import from your existing files
from main_planning import CDFVisualizer
from mlp import MLPRegression
from nn_cdf import CDF

@dataclass
class Bubble:
    center: np.ndarray
    radius: float
    
class BubblePlanner:
    def __init__(self, cdf_visualizer: CDFVisualizer):
        self.visualizer = cdf_visualizer
        self.cdf = cdf_visualizer.cdf
        self.robot_id = cdf_visualizer.robot_id
        self.obstacle_points = cdf_visualizer.obstacle_points
        
        # Planning parameters
        self.epsilon = 5E-2  # Bubble expansion parameter
        self.min_radius = 1E-10
        self.num_samples = 3000
        self.max_iterations = 1000
        self.step_size = 0.1
        
    def sample_config(self) -> np.ndarray:
        """Sample random configuration within joint limits"""
        config = []
        for i in range(7):  # Franka has 7 joints
            lower, upper = p.getJointInfo(self.robot_id, i)[8:10]
            config.append(np.random.uniform(lower, upper))
        return np.array(config)
    
    def get_bubble(self, config: np.ndarray) -> Bubble:
        """Generate a bubble at the given configuration"""
        # Query CDF to get distance to obstacles
        distance = self.visualizer.query_cdf(config)
        return Bubble(center=config, radius=max(distance - self.epsilon, self.min_radius))
    
    def bubbles_overlap(self, b1: Bubble, b2: Bubble) -> bool:
        """Check if two bubbles overlap"""
        center_dist = np.linalg.norm(b1.center - b2.center)
        return center_dist < (b1.radius + b2.radius)
    
    def generate_bubbles(self, start_config: np.ndarray, goal_config: np.ndarray):
        """Generate bubbles using rapidly exploring random tree"""
        # Convert input configs to float32
        start_config = start_config.astype(np.float32)
        goal_config = goal_config.astype(np.float32)
        
        # Wrap the CDF query to ensure float32 inputs
        def sdf(x):
            x = np.array(x, dtype=np.float32)
            return self.visualizer.query_cdf(x)
        
        # Define joint limits (as float32)
        mins = np.array([p.getJointInfo(self.robot_id, i)[8] for i in range(7)], dtype=np.float32)
        maxs = np.array([p.getJointInfo(self.robot_id, i)[9] for i in range(7)], dtype=np.float32)
        
        overlaps_graph, max_circles, _ = get_rapidly_exploring(
            sdf,
            self.epsilon,
            self.min_radius,
            self.num_samples,
            mins,
            maxs,
            start_config,
            end_point=goal_config,
            max_retry=1000,
            max_retry_epsilon=1000,
            max_num_iterations=self.num_samples
        )
        
        return overlaps_graph, max_circles
        
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray):
        """Main planning function"""
        print("Generating bubbles...")
        overlaps_graph, max_circles = self.generate_bubbles(start_config, goal_config)
        
        print("Finding start and end indices...")
        start_idx = position_to_max_circle_idx(overlaps_graph, start_config)
        if start_idx < 0:
            print("Repairing graph for start")
            overlaps_graph, start_idx = trace_toward_graph_all(
                overlaps_graph, 
                lambda x: self.visualizer.query_cdf(x),
                self.epsilon, 
                self.min_radius, 
                start_config
            )
            
        end_idx = position_to_max_circle_idx(overlaps_graph, goal_config)
        if end_idx < 0:
            print("Repairing graph for end")
            overlaps_graph, end_idx = trace_toward_graph_all(
                overlaps_graph, 
                lambda x: self.visualizer.query_cdf(x),
                self.epsilon, 
                self.min_radius, 
                goal_config
            )
            
        print("Finding shortest path...")
        epath_centre_distance = get_shortest_path(
            lambda from_circle, to_circle: from_circle.hausdorff_distance_to(to_circle),
            overlaps_graph,
            start_idx,
            end_idx,
            cost_name="cost",
            return_epath=True,
        )
        
        print("Optimizing path...")
        bps, constr_bps = edgeseq_to_traj_constraint_bezier(
            overlaps_graph.es[epath_centre_distance[0]], 
            start_config, 
            goal_config
        )
        
        cost = bezier_cost_all(bps)
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constr_bps)
        prob.solve(verbose=True)
        
        times = np.linspace(0, 1.0, 50)
        query = np.vstack([bp.query(times).value for bp in bps])
        
        return query, max_circles
    
    def execute_path(self, path: np.ndarray, visualize: bool = True, speed_factor: float = 5.0):
        """Execute the planned path on the robot"""
        path = np.array(path, dtype=np.float32)
        
        for config in path:
            # Ensure config is float32
            config = np.array(config, dtype=np.float32)
            
            # Set joint positions
            for i in range(len(config)):
                p.setJointMotorControl2(
                    self.visualizer.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    float(config[i]),
                    force=500,  # Increased force
                    maxVelocity=speed_factor  # Control execution speed
                )
            
            if visualize:
                min_dist = self.visualizer.query_cdf(config)
                self.visualizer.visualize_distances(min_dist)
            
            p.stepSimulation()
            time.sleep(1./(240. * speed_factor))  # Reduced sleep time

def main():
    # Initialize visualizer
    visualizer = CDFVisualizer()
    
    # Create planner
    planner = BubblePlanner(visualizer)
    
    # Define start and goal configurations
    start_config = np.zeros(7)  # Home position
    goal_config = np.array([0.5, 0.5, 0, -0.5, 0, 0.5, 0])  # Example goal
    
    # Plan path
    path, max_circles = planner.plan(start_config, goal_config)

    print('path', path)
    print('max_circles', max_circles)
    
    if path is not None:
        # Execute path
        planner.execute_path(path)
    
    # Keep simulation running
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main() 