import pybullet as p
import numpy as np
import cvxpy
from dataclasses import dataclass
from sdf_marching.samplers import get_rapidly_exploring, get_uniform_random, get_rapidly_exploring_connect
from sdf_marching.overlap import position_to_max_circle_idx
from sdf_marching.samplers.tracing import trace_toward_graph_all
from sdf_marching.discrete import get_shortest_path
from sdf_marching.cvx import edgeseq_to_traj_constraint_bezier, bezier_cost_all
import cProfile

@dataclass
class Bubble:
    center: np.ndarray
    radius: float
    
class BubblePlanner:
    def __init__(self, cdf_visualizer, random_seed=42):
        # Store the random seed as instance variable
        self.random_seed = random_seed
        
        self.visualizer = cdf_visualizer
        self.cdf = cdf_visualizer.cdf
        self.robot_id = cdf_visualizer.robot_id
        self.obstacle_points = cdf_visualizer.obstacle_points
        
        # Planning parameters
        self.epsilon = 5E-2  # Bubble expansion parameter
        self.min_radius = 1E-2
        self.num_samples = 5E4
        self.max_iterations = 5E4
        self.step_size = 0.2
        
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
        """Generate bubbles using uniform random sampling"""
        
        # Convert input configs to float32 and ensure correct shape
        start_config = start_config.astype(np.float32)[:7]
        goal_config = goal_config.astype(np.float32)[:7]
        
        # Wrap the CDF query to ensure float32 inputs and correct shape
        def cdf(x):
            
            if len(x.shape) > 1:
                # Handle batch input
                results = []
                for single_x in x:
                    single_x = single_x[:7].astype(np.float32)
                    value = self.visualizer.query_cdf(single_x)
                    results.append(value)
                return np.array(results)
            else:
                # Handle single input
                x = x[:7].astype(np.float32)
                return self.visualizer.query_cdf(x)
        
        # Define joint limits (as float32)
        mins = np.array([p.getJointInfo(self.robot_id, i)[8] for i in range(7)], dtype=np.float32)
        maxs = np.array([p.getJointInfo(self.robot_id, i)[9] for i in range(7)], dtype=np.float32)
        
        try:
            # Create random number generator with our seed
            rng = np.random.default_rng(self.random_seed)
            
            # Method 1: RRT-based bubble generation
            overlaps_graph, max_circles, _ = get_rapidly_exploring_connect(
                cdf,
                self.epsilon,
                self.min_radius,
                int(self.num_samples),
                mins,
                maxs,
                start_point=start_config,
                batch_size=100,
                max_retry=500,
                max_retry_epsilon=100,
                max_num_iterations=int(self.max_iterations),
                inflate_factor=1.0,
                prc=0.1,
                end_point=goal_config,
                rng=rng  # Pass the seeded random number generator
            )
            
            # Method 2: Uniform random sampling
            # print("Starting uniform random sampling...")
            # overlaps_graph, max_circles = get_uniform_random(
            #     cdf,
            #     self.epsilon,
            #     self.min_radius,
            #     self.num_samples,  # Convert to int explicitly
            #     mins,
            #     maxs,
            #     start_point=start_config
            # )
            
            print(f"\nInitial bubble generation complete:")
            print(f"Number of bubbles in graph: {len(overlaps_graph.vs)}")
            print(f"Number of max circles: {len(max_circles)}")
            return overlaps_graph, max_circles
            
        except Exception as e:
            print(f"Bubble generation failed after creating {len(max_circles) if 'max_circles' in locals() else 0} bubbles")
            print(f"Error details: {str(e)}")
            raise e

    def plan(self, start_config: np.ndarray, goal_config: np.ndarray):
        """Main planning function"""
        print("\nDEBUG: Starting planning process...")
        try:
            #with cProfile.Profile() as pr:
            overlaps_graph, max_circles = self.generate_bubbles(start_config, goal_config)
            #pr.print_stats('cumtime')
            
            # Print number of bubbles created
            num_bubbles = len(max_circles)
            print(f"\nNumber of bubbles created: {num_bubbles}")
            
            start_idx = position_to_max_circle_idx(overlaps_graph, start_config)
            print(f"Start index: {start_idx}")
            
            if start_idx < 0:
                print("Connecting start config to graph...")
                overlaps_graph, start_idx = trace_toward_graph_all(
                    overlaps_graph, 
                    lambda x: self.visualizer.query_cdf(x),
                    self.epsilon, 
                    self.min_radius, 
                    start_config
                )
                print(f"Number of bubbles after connecting start: {len(overlaps_graph.vs)}")
            
            end_idx = position_to_max_circle_idx(overlaps_graph, goal_config)
            print(f"End index: {end_idx}")
            
            # if end_idx < 0:
            #     print("Connecting goal config to graph...")
            #     overlaps_graph, end_idx = trace_toward_graph_all(
            #         overlaps_graph, 
            #         lambda x: self.visualizer.query_cdf(x),
            #         self.epsilon, 
            #         self.min_radius, 
            #         goal_config
            #     )
            #     print(f"Number of bubbles after connecting goal: {len(overlaps_graph.vs)}")
            
            
            # Store original goal config
            actual_goal_config = goal_config.copy()
            
            # If end_idx is negative, find closest bubble
            if end_idx < 0:
                # Get all bubble centers
                centers = np.array([v['circle'].centre for v in overlaps_graph.vs])
                distances_to_goal = np.linalg.norm(centers - goal_config, axis=1)
                end_idx = np.argmin(distances_to_goal)
                print(f"No direct connection to goal. Planning to closest bubble at distance {distances_to_goal[end_idx]:.4f}")
                # Use the center of the closest bubble as our actual endpoint
                print('End index:', end_idx)
                actual_goal_config = centers[end_idx].copy()
            
            # Convert graph to directed for path planning
            overlaps_graph.to_directed()
            
            # Find shortest path
            path_result = get_shortest_path(
                lambda from_circle, to_circle: from_circle.hausdorff_distance_to(to_circle),
                overlaps_graph,
                start_idx,
                end_idx,
                cost_name="cost",
                return_epath=True,
            )
            
            if isinstance(path_result, float):
                raise ValueError("Could not find path to closest reachable point")
            
            # Generate trajectory
            try:
                bps, constr_bps = edgeseq_to_traj_constraint_bezier(
                    overlaps_graph.es[path_result[0]], 
                    start_config, 
                    actual_goal_config  # Use the actual endpoint we're planning to
                )
                
                cost = bezier_cost_all(bps)
                prob = cvxpy.Problem(cvxpy.Minimize(cost), constr_bps)
                prob.solve(verbose=True)
                
                times = np.linspace(0, 1.0, 50)
                query = np.vstack([bp.query(times).value for bp in bps])
                
                return query, max_circles
                
            except Exception as e:
                print(f"Failed to generate trajectory: {str(e)}")
                raise e
                
        except Exception as e:
            print(f"Planning failed with error: {str(e)}")
            raise e
    
    
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

    return None

if __name__ == "__main__":
    main() 