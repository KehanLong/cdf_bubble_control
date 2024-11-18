import pybullet as p
import numpy as np
import time
from main_planning import CDFVisualizer

class IKSolver:
    def __init__(self):
        # Initialize PyBullet and load robot
        self.visualizer = CDFVisualizer()
        self.robot_id = self.visualizer.robot_id
        
    def verify_ik_solution(self, config, target_pos, tolerance=0.01):
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
        ee_state = p.getLinkState(self.robot_id, 9)  # end effector link
        achieved_pos = ee_state[0]
        
        # Calculate error
        error = np.linalg.norm(np.array(achieved_pos) - np.array(target_pos))
        
        # Reset robot to original position
        for i in range(7):
            p.resetJointState(self.robot_id, i, float(original_positions[i]))
        
        return error < tolerance, error, achieved_pos

    def is_within_limits(self, config):
        """Check if configuration is within joint limits"""
        for i in range(7):
            info = p.getJointInfo(self.robot_id, i)
            if config[i] < info[8] or config[i] > info[9]:
                return False
        return True

    def normalize_angles(self, config):
        """Normalize angles to be within joint limits"""
        normalized_config = []
        for i in range(7):
            angle = config[i]
            info = p.getJointInfo(self.robot_id, i)
            lower, upper = info[8], info[9]
            
            # Normalize angle to [-pi, pi]
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            
            # Clamp to joint limits
            angle = np.clip(angle, lower, upper)
            normalized_config.append(angle)
        
        return np.array(normalized_config)

    def find_ik_solutions(self, target_pos, num_solutions=5):
        """Find multiple IK solutions for a target position"""
        print(f"\nSearching for IK solutions to reach target: {target_pos}")
        
        # First, print current robot state
        current_ee_state = p.getLinkState(self.robot_id, 9)
        print(f"Current end effector position: {current_ee_state[0]}")
        print(f"Current end effector orientation: {current_ee_state[1]}")
        
        # Print joint limits
        print("\nJoint limits:")
        for i in range(7):
            info = p.getJointInfo(self.robot_id, i)
            print(f"Joint {i}: [{info[8]:.2f}, {info[9]:.2f}]")
        
        valid_solutions = []
        attempts = 0
        max_attempts = 1000
        
        while len(valid_solutions) < num_solutions and attempts < max_attempts:
            # Try different initial configurations
            if attempts % 100 == 0:
                print(f"\nAttempt {attempts}/{max_attempts}")
            
            # Randomize orientation more
            noise = np.random.normal(0, 0.5, 3)  # Increased noise
            noisy_orientation = p.getQuaternionFromEuler(noise)
            
            # Try IK with different parameters
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                9,  # end effector link
                target_pos,
                noisy_orientation,
                maxNumIterations=2000,      # Increased iterations
                residualThreshold=0.001,    # Keep small threshold
                jointDamping=[0.1]*10,      # Damping
                lowerLimits=[p.getJointInfo(self.robot_id, i)[8] for i in range(7)],
                upperLimits=[p.getJointInfo(self.robot_id, i)[9] for i in range(7)],
                restPoses=[0.0] * 7  # Try neutral pose as rest pose
            )
            
            if joint_poses is not None:
                config = np.array(joint_poses[:7], dtype=np.float32)
                
                # Normalize angles and check limits
                config = self.normalize_angles(config)
                if not self.is_within_limits(config):
                    continue
                
                # Refine IK solution
                for _ in range(5):
                    # Get current end effector position
                    for j in range(7):
                        p.resetJointState(self.robot_id, j, float(config[j]))
                    ee_state = p.getLinkState(self.robot_id, 9)
                    current_pos = ee_state[0]
                    
                    # Calculate error
                    error_vec = np.array(target_pos) - np.array(current_pos)
                    error = np.linalg.norm(error_vec)
                    
                    if error < 0.01:  # If within tolerance, break
                        break
                    
                    # Try to improve solution with small random adjustments
                    delta = np.random.normal(0, 0.01, 7)  # Small random changes
                    new_config = config + delta
                    
                    # Ensure new config is within limits
                    new_config = self.normalize_angles(new_config)
                    if not self.is_within_limits(new_config):
                        continue
                    
                    # Check if new config is better
                    for j in range(7):
                        p.resetJointState(self.robot_id, j, float(new_config[j]))
                    new_ee_state = p.getLinkState(self.robot_id, 9)
                    new_error = np.linalg.norm(np.array(target_pos) - np.array(new_ee_state[0]))
                    
                    if new_error < error:
                        config = new_config
                        error = new_error
                
                # Verify final solution
                is_valid, error, achieved_pos = self.verify_ik_solution(config, target_pos)
                
                if is_valid and self.is_within_limits(config):  # Double check limits
                    valid_solutions.append({
                        'config': config,
                        'error': error,
                        'achieved_pos': achieved_pos
                    })
                    print(f"\nFound valid solution {len(valid_solutions)}:")
                    print(f"Configuration: {config}")
                    print(f"Error: {error:.4f} meters")
                    print(f"Achieved position: {achieved_pos}")
            
            attempts += 1
        
        if len(valid_solutions) == 0:
            print("\nNo valid IK solutions found!")
            print("Last attempted configuration details:")
            print(f"Target position: {target_pos}")
            print(f"Last achieved position: {achieved_pos}")
            print(f"Error: {error:.4f} meters")
        else:
            print(f"\nFound {len(valid_solutions)} valid solutions in {attempts} attempts")
        
        return valid_solutions

def main():
    solver = IKSolver()
    
    # Define target position
    target_pos = [0.4, 0.5, 0.6]  # Example target position
    
    # Find IK solutions
    solutions = solver.find_ik_solutions(target_pos)
    
    # Visualize each solution
    if solutions:
        print("\nPress Enter to visualize each solution...")
        for i, sol in enumerate(solutions):
            input(f"\nPress Enter to see solution {i+1}")
            config = sol['config']
            for j in range(7):
                p.resetJointState(solver.robot_id, j, float(config[j]))
            time.sleep(1)  # Give time to visualize
    
    # Keep simulation running
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 