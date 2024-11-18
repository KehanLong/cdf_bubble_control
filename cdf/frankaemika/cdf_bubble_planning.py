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
        # Convert input configs to float32 and ensure correct shape
        start_config = start_config.astype(np.float32)[:7]  # Take only first 7 joints
        goal_config = goal_config.astype(np.float32)[:7]    # Take only first 7 joints
        
        # Wrap the CDF query to ensure float32 inputs and correct shape
        def cdf(x):
            # If x is 2D array, take first 7 elements of each row
            if len(x.shape) > 1:
                x = x[:, :7]
            else:
                # If x is 1D array, take first 7 elements
                x = x[:7]
            x = np.array(x, dtype=np.float32)
            return self.visualizer.query_cdf(x)
        
        # Define joint limits (as float32)
        mins = np.array([p.getJointInfo(self.robot_id, i)[8] for i in range(7)], dtype=np.float32)
        maxs = np.array([p.getJointInfo(self.robot_id, i)[9] for i in range(7)], dtype=np.float32)
        
        overlaps_graph, max_circles, _ = get_rapidly_exploring(
            cdf,
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
    
    
    def execute_path(self, path: np.ndarray, visualize: bool = True, speed_factor: float = 0.1):
        """Execute the planned path on the robot and visualize CDF values"""
        path = np.array(path, dtype=np.float32)
        
        print(f"Executing path with {len(path)} waypoints")
        print(f"Start config: {path[0]}")
        print(f"Goal config: {path[-1]}")
        
        # For recording CDF values
        cdf_values = []
        time_steps = []
        start_time = time.time()
        
        # Create line for trajectory visualization
        ee_positions = []
        
        for i, config in enumerate(path):
            # Print progress every 10 steps
            if i % 10 == 0:
                print(f"Step {i}/{len(path)}")
            
            # Set joint positions directly
            for joint_idx, joint_val in enumerate(config):
                p.resetJointState(self.robot_id, joint_idx, float(joint_val))
            
            # Step simulation a few times to stabilize
            for _ in range(10):
                p.stepSimulation()
            
            # Get and draw end effector position
            ee_state = p.getLinkState(self.robot_id, 9)  
            ee_pos = ee_state[0]  # Position in world frame
            ee_positions.append(ee_pos)
            
            # Draw trajectory line
            if len(ee_positions) >= 2:
                p.addUserDebugLine(
                    ee_positions[-2],
                    ee_positions[-1],
                    lineColorRGB=[0, 0, 1],  # Blue color
                    lineWidth=2.0,
                    lifeTime=0  # Permanent until removed
                )
            
            # Record CDF value
            cdf_value = self.visualizer.query_cdf(config)
            current_time = time.time() - start_time
            cdf_values.append(cdf_value)
            time_steps.append(current_time)
            
            # Visualize current CDF value
            #self.visualizer.visualize_distances(cdf_value)
            
            # Small delay between waypoints
            time.sleep(0.1)
        
        print("Reached goal configuration")
        
        # Plot CDF values
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.plot(time_steps, cdf_values, 'b-', label='CDF Value')
            plt.xlabel('Time (s)')
            plt.ylabel('Minimum Distance to Obstacles (m)')
            plt.title('CDF Values Along Planned Trajectory')
            plt.grid(True)
            plt.legend()
            plt.savefig('cdf_values.png', dpi=300)
        except ImportError:
            print("Matplotlib not available for plotting")
        
        return cdf_values, time_steps
    
    def verify_ik_solution(self, config, target_pos, tolerance=0.05):
        """
        Verify if a configuration reaches the target position
        Args:
            config: joint configuration to test
            target_pos: desired end-effector position
            tolerance: maximum allowed distance error (in meters)
        """
        # Store current joint positions
        original_positions = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        # Set robot to test configuration
        for i in range(7):
            p.resetJointState(self.robot_id, i, float(config[i]))
        
        # Get end effector position through forward kinematics
        ee_state = p.getLinkState(self.robot_id, 9)  # Changed to link 9 (end effector)
        achieved_pos = ee_state[0]
        
        # Calculate error
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
        
        # Reset robot to original position
        for i in range(7):
            p.resetJointState(self.robot_id, i, float(original_positions[i]))
        
        print("\nIK Verification:")
        print(f"Target position: {target_pos}")
        print(f"Achieved position: {achieved_pos}")
        print(f"Position error: {error:.4f} meters")
        print(f"Within tolerance: {error < tolerance}")
        
        return error < tolerance

    def find_safe_goal_config(self, target_pos, num_ik_solutions=100, safety_threshold=0.1):
        """Find a safe goal configuration that reaches the target position"""
        ik_solutions = []
        
        print(f"\nSearching for IK solutions to reach target: {target_pos}")
        
        # Current end effector position for orientation reference
        current_ee_state = p.getLinkState(self.robot_id, 9)  # Changed to link 9 (end effector)
        current_orientation = current_ee_state[1]
        
        # Try multiple IK solutions with better parameters
        for i in range(num_ik_solutions):
            # Add small random noise to orientation to get different IK solutions
            noise = np.random.normal(0, 0.1, 3)
            noisy_orientation = p.getQuaternionFromEuler(noise)
            
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                9,  # Changed to link 9 (end effector)
                target_pos,
                noisy_orientation,
                maxNumIterations=1000,    # Increased iterations
                residualThreshold=0.005,  # Reduced threshold
                jointDamping=[0.1]*10      # Added joint damping
            )
            
            if joint_poses is not None:
                # Only take first 7 joints
                config = np.array(joint_poses[:7], dtype=np.float32)
                
                # Verify IK solution with larger tolerance
                if self.verify_ik_solution(config, target_pos, tolerance=0.1):  # 5cm tolerance
                    ik_solutions.append(config)
                    print(f"Found valid IK solution {len(ik_solutions)}")
        
        print(f"\nFound {len(ik_solutions)} valid IK solutions")
        
        # Evaluate safety of each solution
        safe_configs = []
        safety_scores = []
        
        for config in ik_solutions:
            safety_score = self.visualizer.query_cdf(config)
            
            if safety_score > safety_threshold:
                safe_configs.append(config)
                safety_scores.append(safety_score)
                print(f"Found safe configuration with score: {safety_score:.4f}")
        
        if not safe_configs:
            raise ValueError("No safe IK solutions found!")
        
        # Choose the safest configuration
        best_idx = np.argmax(safety_scores)
        best_config = safe_configs[best_idx]
        
        print(f"\nSelected best configuration:")
        print(f"Safety score: {safety_scores[best_idx]:.4f}")
        print(f"Configuration: {best_config}")
        
        # Final verification of best solution
        print("\nFinal verification of selected configuration:")
        self.verify_ik_solution(best_config, target_pos)
        
        return best_config

    def plan_to_position(self, target_pos):
        """Plan a path to reach a target end-effector position"""
        # Get current configuration as start, but only take first 7 joints
        start_config = np.array([p.getJointState(self.robot_id, i)[0] for i in range(7)])  # Only get 7 joints
        
        # Find a safe goal configuration
        try:
            goal_config = self.find_safe_goal_config(target_pos)
        except ValueError as e:
            print(f"Planning failed: {e}")
            return None, None
        
        print("Start config:", start_config)  # Debug print
        print("Goal config:", goal_config)    # Debug print
        
        # Ensure both configs are float32 and 7D
        start_config = np.array(start_config, dtype=np.float32)[:7]
        goal_config = np.array(goal_config, dtype=np.float32)[:7]
        
        # Plan path using existing method
        return self.plan(start_config, goal_config)

def main():
    # Initialize visualizer
    visualizer = CDFVisualizer()
    planner = BubblePlanner(visualizer)
    
    # Choose whether to plan with target position or configuration
    use_target_pos = False  # Set to False to use direct configuration
    
    if use_target_pos:
        # Define target position for end effector
        target_pos = [0.4, 0.5, 0.6]  # Example target position
        print(f"Planning path to target position: {target_pos}")
        path, max_circles = planner.plan_to_position(target_pos)
    else:
        # Define target configuration directly
        target_config = np.array([1.68431763,  0.29743382, -0.65842076 ,-1.87699534, -2.26396217,  1.34391705,
                                   0.20779162], dtype=np.float32)
        
        # Get current configuration as start
        start_config = np.array([p.getJointState(planner.robot_id, i)[0] for i in range(7)])
        
        print("Planning path with direct configuration:")
        print(f"Start config: {start_config}")
        print(f"Goal config: {target_config}")
        
        path, max_circles = planner.plan(start_config, target_config)
    
    if path is not None:
        print("\nPath Statistics:")
        print(f"Number of waypoints: {len(path)}")
        print(f"Start config: {path[0]}")
        print(f"End config: {path[-1]}")
        
        # Execute path
        planner.execute_path(path)
    else:
        print("Planning failed!")

if __name__ == "__main__":
    main() 