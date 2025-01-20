try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    print("Warning: OMPL could not be imported. Some functionality may be limited.")
    

import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class PlanningMetrics:
    success: bool
    num_collision_checks: int
    path_length: float
    num_samples: int
    planning_time: float
    reached_goal_index: int

def to_tensor(s, dof):
    return torch.tensor([s[i] for i in range(dof)], dtype=torch.float32)

class ValidityCheckerWrapper:
    def __init__(self, robot_sdf, robot_cdf, points_robot, dof=6, safety_margin=0.01, device='cuda'):
        self.robot_sdf = robot_sdf
        self.robot_cdf = robot_cdf
        self.points_robot = points_robot  # Should be shape [N, 3]
        self.device = device
        self.safety_margin = safety_margin
        self.dof = dof
        self.counter = 0
    
    def __call__(self, state) -> bool:
        """Check if configuration is collision-free"""
        self.counter += 1
        
        # Convert state to tensor [1, 6] (batch size 1)
        config = torch.tensor([state[i] for i in range(self.dof)], 
                            device=self.device, 
                            dtype=torch.float32).unsqueeze(0)
        
        # Reshape points to [1, N, 3] (batch size 1)
        points = self.points_robot.unsqueeze(0)  # Add batch dimension
        
        # Debug prints
        # print("\nDebug SDF Query:")
        # print(f"Points shape: {points.shape}")
        # print(f"Config shape: {config.shape}")
        
        # Query SDF values
        sdf_values = self.robot_sdf.query_sdf(
            points=points,  # Shape: [1, N, 3]
            joint_angles=config  # Shape: [1, num_links]
        )
        
        # Debug prints for SDF values
        # print(f"SDF values shape: {sdf_values.shape}")
        # print(f"SDF min value: {sdf_values.min().item():.6f}")
        # print(f"SDF max value: {sdf_values.max().item():.6f}")
        # print(f"SDF mean value: {sdf_values.mean().item():.6f}")
        
        is_valid = sdf_values.min().item() > self.safety_margin
        #print(f"Checking config {config.cpu().numpy()[0]}, valid: {is_valid}")
        return is_valid
    
    def reset_count(self):
        self.counter = 0

class OMPLRRTPlanner:
    def __init__(self, 
                 robot_sdf,
                 robot_cdf,
                 robot_fk,
                 joint_limits: tuple,
                 planner_type: str = 'rrt',
                 device: str = 'cuda',
                 seed: int = None,
                 safety_margin: float = 0.01):
        """
        Initialize OMPL RRT planner
        
        Args:
            robot_sdf: Robot SDF model
            robot_fk: Robot forward kinematics
            joint_limits: Tuple of (lower_limits, upper_limits)
            planner_type: 'rrt', 'rrt_connect', or 'rrt_star'
            device: Computing device
            seed: Random seed for reproducibility
        """
        self.robot_sdf = robot_sdf
        self.robot_cdf = robot_cdf
        self.robot_fk = robot_fk
        self.device = device
        self.dof = len(joint_limits[0])
        self.safety_margin = safety_margin
        # Set random seed if provided
        if seed is not None:
            print(f"Setting RRT random seed: {seed}")
            ou.RNG.setSeed(seed)  # Set OMPL's random seed
        
        # Setup state space
        self.space = ob.RealVectorStateSpace(self.dof)
        bounds = ob.RealVectorBounds(self.dof)
        for i, (low, high) in enumerate(zip(joint_limits[0], joint_limits[1])):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.space.setBounds(bounds)
        
        # Setup space information
        self.si = ob.SpaceInformation(self.space)
        
        # Will be set during planning
        self.validity_checker = None
        self.planner_type = planner_type
        
    def plan(self, 
             start_config: np.ndarray,
             goal_configs: List[np.ndarray],
             obstacle_points: torch.Tensor,
             max_time: float = 30.0,
             return_metrics: bool = True) -> Dict[str, Any]:
        """
        Plan path using OMPL RRT with multiple goals
        """
        print("\nPlanning Debug Info:")
        print(f"Start config: {start_config}")
        print(f"Number of goals: {len(goal_configs)}")
        
        # Setup validity checker
        self.validity_checker = ValidityCheckerWrapper(self.robot_sdf, self.robot_cdf, obstacle_points, 
                                                       self.dof, self.safety_margin, self.device)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.validity_checker))

        check_resolution = 0.01    # 0.01 for 2 joints, 0.001 for 6 joints
        self.si.setStateValidityCheckingResolution(check_resolution)
        self.si.setup()
        
        # Setup problem definition
        pdef = ob.ProblemDefinition(self.si)
        
        # Create start state
        start = ob.State(self.space)
        for i in range(self.dof):
            start[i] = float(start_config[i])
        pdef.addStartState(start)
        
        # Create goal states and add them as multiple goals
        goal_threshold = 0.05
        print(f"Using goal threshold: {goal_threshold}")
        
        # Create a MultiGoalRegion
        mg = ob.GoalStates(self.si)
        
        # Add each goal state to the MultiGoalRegion
        for i, goal_config in enumerate(goal_configs):
            goal = ob.State(self.space)
            for j in range(self.dof):
                goal[j] = float(goal_config[j])
            mg.addState(goal)
            print(f"Added goal {i}: {goal_config}")
        
        # Set the multi-goal as the problem goal
        pdef.setGoal(mg)
        
        # Create and setup planner
        planner = og.RRT(self.si)
        planner.setRange(0.1)
        planner.setGoalBias(0.1)
        print(f"RRT parameters - Step size: {planner.getRange()}, Goal bias: {planner.getGoalBias()}")
        
        planner.setProblemDefinition(pdef)
        planner.setup()
        
        # Solve
        start_time = time.time()
        solved = planner.solve(max_time)
        planning_time = time.time() - start_time
        
        print(f"\nPlanning solved: {solved}")
        
        # Extract solution and metrics
        if solved:
            # Get path
            path = pdef.getSolutionPath()
            waypoints = []
            for state in path.getStates():
                waypoint = [state[i] for i in range(self.dof)]
                waypoints.append(waypoint)
            waypoints = np.array(waypoints)
            
            print(f"\nPath found with {len(waypoints)} waypoints")
            # print("First few waypoints:")
            # for i in range(min(3, len(waypoints))):
            #     print(f"Waypoint {i}: {waypoints[i]}")
            # print("Last few waypoints:")
            # for i in range(max(0, len(waypoints)-2), len(waypoints)):
            #     print(f"Waypoint {i}: {waypoints[i]}")
            
            # Find which goal was reached
            final_waypoint = waypoints[-1]
            reached_goal_index = 0
            min_dist = float('inf')
            for i, goal_config in enumerate(goal_configs):
                dist = np.linalg.norm(final_waypoint - goal_config)
                if dist < min_dist:
                    min_dist = dist
                    reached_goal_index = i
            
            print(f"\nReached goal index: {reached_goal_index}")
            print(f"Final distance to reached goal: {min_dist}")
            
            # Compute path length
            path_length = 0.0
            for i in range(len(waypoints)-1):
                path_length += np.linalg.norm(waypoints[i+1] - waypoints[i])
            
            # Get number of vertices using PlannerData
            pdata = ob.PlannerData(self.si)
            planner.getPlannerData(pdata)
            num_vertices = pdata.numVertices()
            
            metrics = PlanningMetrics(
                success=True,
                num_collision_checks=self.validity_checker.counter,  # Total checks (vertex + edge)
                path_length=path_length,
                num_samples=num_vertices,  # Using PlannerData to get vertex count
                planning_time=planning_time,
                reached_goal_index=reached_goal_index
            )
            
            return {
                'waypoints': waypoints,
                'metrics': metrics
            }
        else:
            metrics = PlanningMetrics(
                success=False,
                num_collision_checks=self.validity_checker.counter,
                path_length=float('inf'),
                num_samples=self.validity_checker.counter,
                planning_time=planning_time,
                reached_goal_index=-1
            )
            
            return {
                'waypoints': None,
                'metrics': metrics
            }

def test_ompl_planner():
    """Test function"""
    import torch
    # Add project root to Python path
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    from xarm_planning import XArmSDFVisualizer
    
    # Initialize visualizer with a dummy goal
    goal_pos = torch.tensor([0.7, 0.2, 0.6], device='cuda')
    visualizer = XArmSDFVisualizer(goal_pos, use_gui=False)
    
    # Create OMPL planner
    planner = OMPLRRTPlanner(
        robot_sdf=visualizer.robot_sdf,
        robot_fk=visualizer.robot_fk,
        joint_limits=(
            visualizer.robot_fk.joint_limits[:, 0].cpu().numpy(),
            visualizer.robot_fk.joint_limits[:, 1].cpu().numpy()
        ),
        planner_type='rrt'
    )
    
    # Test planning
    start_config = np.zeros(6)
    goal_configs = [np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])]
    
    result = planner.plan(
        start_config=start_config,
        goal_configs=goal_configs,
        obstacle_points=visualizer.points_robot
    )
    
    if result['metrics'].success:
        print("Planning succeeded!")
        print(f"Path length: {result['metrics'].path_length:.3f}")
        print(f"Collision checks: {result['metrics'].num_collision_checks}")
        print(f"Number of samples: {result['metrics'].num_samples}")
        print(f"Planning time: {result['metrics'].planning_time:.3f}s")
    else:
        print("Planning failed!")

if __name__ == "__main__":
    test_ompl_planner() 