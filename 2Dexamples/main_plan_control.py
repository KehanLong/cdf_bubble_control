import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main_control import track_planned_path
from main_planning import plan_and_visualize
from utils_env import create_obstacles, create_animation
from robot_sdf import RobotSDF
from robot_cdf import RobotCDF
from control.reference_governor_bezier import BezierReferenceGovernor

class PlanningAndControlManager:
    def __init__(self, 
                 robot_cdf: RobotCDF,
                 robot_sdf: RobotSDF,
                 obstacles: List[np.ndarray],
                 control_type: str = 'clf_dro_cbf',
                 planner_type: str = 'bubble',
                 dt: float = 0.02,
                 stuck_threshold: float = 0.1,
                 progress_window: int = 50,
                 replan_lookahead: int = 5):
        """
        Initialize the planning and control manager.
        
        Args:
            stuck_threshold: Threshold for detecting when robot is stuck
            progress_window: Number of steps to monitor progress
            replan_lookahead: Number of waypoints to look ahead for local goal
        """
        self.robot_cdf = robot_cdf
        self.robot_sdf = robot_sdf
        self.obstacles = obstacles
        self.control_type = control_type
        self.planner_type = planner_type
        self.dt = dt
        
        self.stuck_threshold = stuck_threshold
        self.progress_window = progress_window
        self.replan_lookahead = replan_lookahead
        
        self.device = robot_sdf.device

    def is_robot_stuck(self, 
                      recent_configs: List[np.ndarray],
                      recent_controls: List[np.ndarray],
                      current_s: float,
                      previous_s: float) -> bool:
        """Check if robot is stuck based on multiple criteria."""
        if len(recent_configs) < self.progress_window:
            return False
            
        # Criterion 1: Check if control inputs are very small
        recent_control_magnitudes = [np.linalg.norm(u) for u in recent_controls[-10:]]
        if np.mean(recent_control_magnitudes) < self.stuck_threshold:
            print('control magnitudes', recent_control_magnitudes)
            print("Stuck detected: Control inputs too small")
            return True
            
        # # Criterion 2: Check if robot is making progress along the path
        # if abs(current_s - previous_s) < 0.001:  # Minimal progress in path parameter
        #     print("Stuck detected: No progress along path")
        #     return True
            
        # # Criterion 3: Check if position has changed significantly
        # recent_movement = np.linalg.norm(
        #     recent_configs[-1] - recent_configs[-self.progress_window]
        # )
        # if recent_movement < self.stuck_threshold:
        #     print("Stuck detected: Minimal physical movement")
        #     return True
            
        return False

    def get_local_goal(self, 
                      current_config: np.ndarray,
                      governor: BezierReferenceGovernor,
                      current_s: float) -> Tuple[np.ndarray, float]:
        """
        Select an appropriate local goal along the Bezier curve.
        Returns:
            Tuple of (local_goal, goal_s)
        """
        # Look ahead along the path by a certain amount
        goal_s = min(current_s + 0.2, 1.0)  # Look ahead by 20% of the path

        segment_idx, local_s = governor.get_segment_and_local_s(goal_s)
        current_curve = governor.bezier_curves[segment_idx]
        
        local_goal = current_curve.query(local_s).value
        
        return local_goal, goal_s

    def execute_with_replanning(self,
                              initial_config: np.ndarray,
                              goal_configs: np.ndarray,
                              duration: float = 40.0) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Execute the full planning and control loop with replanning capability."""
        
        # Initial planning
        trajectory_data = plan_and_visualize(
            self.robot_cdf, self.robot_sdf, self.obstacles,
            initial_config, goal_configs,
            planner_type=self.planner_type
        )
        
        if trajectory_data is None:
            print("Initial planning failed!")
            return None, None, False

        # Initialize tracking
        current_config = initial_config
        tracked_configs = [current_config]
        reference_configs = [current_config]  
        recent_controls = []
        time = 0
        previous_s = 0
        
        # Initialize reference governor
        governor = BezierReferenceGovernor(
            initial_state=initial_config,
            trajectory_data=trajectory_data,
            dt=self.dt
        )
        
        while time < duration:
            # Get reference from governor
            reference_config, current_s, reference_vel = governor.update(current_config)
            
            # Execute control for a short horizon
            local_tracked_configs, local_reference_configs, tracked_vels, reference_vels, is_safe = track_planned_path(
                self.obstacles,
                trajectory_data,
                current_config,
                dt=self.dt,
                duration=30.0,  # Short control horizon
                control_type=self.control_type,
                use_bezier=True
            )
            
            # Update states and tracking history
            tracked_configs.extend(local_tracked_configs[1:])
            reference_configs.extend(local_reference_configs[1:])
            recent_controls.extend(tracked_vels[1:])
            
            # Get current state
            current_config = local_tracked_configs[-1]
            
            # Check if stuck
            if self.is_robot_stuck(tracked_configs, recent_controls, current_s, previous_s):
                print("\nInitiating local replanning...")
                
                # Get local goal
                local_goal, goal_s = self.get_local_goal(
                    current_config,
                    governor,
                    current_s
                )

                print('-----------------------------------')
                print('local goal', local_goal)
                print('goal_s', goal_s)
                print('-----------------------------------')
                
                # Attempt local replanning
                local_trajectory = plan_and_visualize(
                    self.robot_cdf, self.robot_sdf, self.obstacles,
                    current_config, local_goal[None],  # Add batch dimension
                    planner_type=self.planner_type,
                    max_bubble_samples=50  # Smaller search for faster replanning
                )
                
                if local_trajectory is not None:
                    print("Local replanning successful!")
                    # Combine local replanned Bezier curves with remaining global curves
                    current_segment = int(goal_s * len(trajectory_data['bezier_curves']))
                    new_bezier_curves = (
                        local_trajectory['bezier_curves'] + 
                        trajectory_data['bezier_curves'][current_segment+1:]
                    )
                    trajectory_data['bezier_curves'] = new_bezier_curves
                    
                    # Reinitialize governor with new trajectory
                    governor = BezierReferenceGovernor(
                        initial_state=current_config,
                        trajectory_data=trajectory_data,
                        dt=self.dt
                    )
                else:
                    print("Local replanning failed!")
            
            previous_s = current_s
            time += 1.0  # Increment by control horizon
            
            # Check if goal is reached
            if np.linalg.norm(current_config - goal_configs[0]) < 0.05:
                print("\nGoal reached successfully!")
                break
        
        success = np.linalg.norm(current_config - goal_configs[0]) < 0.05
        if not success:
            print("\nTime limit reached or planning failed!")
        
        return np.array(tracked_configs), np.array(reference_configs), success

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    rng = np.random.default_rng(seed)
    
    # Initialize environment and robots
    obstacles = create_obstacles(rng=rng)
    robot_cdf = RobotCDF(device=device)
    robot_sdf = RobotSDF(device=device)
    
    # Initialize manager
    manager = PlanningAndControlManager(
        robot_cdf=robot_cdf,
        robot_sdf=robot_sdf,
        obstacles=obstacles,
        control_type='clf_dro_cbf',
        planner_type='bubble'
    )
    
    # Execute planning and control
    initial_config = np.array([0., 0.], dtype=np.float32)
    goal_configs = np.array([[np.pi/2, -np.pi/4]], dtype=np.float32)  # Example goal
    
    tracked_configs, reference_configs, success = manager.execute_with_replanning(
        initial_config, goal_configs
    )
    
    # Create animation regardless of success
    if tracked_configs is not None and reference_configs is not None:
        create_animation(obstacles, tracked_configs, reference_configs, 
                        dt=0.02, dynamic_obstacles=True)
    
    return success

if __name__ == "__main__":
    main() 