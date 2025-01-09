import pybullet as p
import numpy as np
import time
from xarm_sim_env import XArmEnvironment

def test_constant_velocity():
    """Test robot movement with constant velocity commands"""
    # Initialize environment
    env = XArmEnvironment(gui=True)
    
    # Set constant velocity for each joint
    constant_velocities = np.array([0.5, 0.2, 0.2, 0.3, 0.2, 0.1])
    
    try:
        print("Applying constant velocities:", constant_velocities)
        
        for _ in range(1000):  # Run for 1000 steps
            # Apply velocity commands
            for i in range(6):
                p.setJointMotorControl2(
                    bodyUniqueId=env.robot_id,
                    jointIndex=i+1,  # Joint indices start from 1
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=constant_velocities[i],
                    force=100
                )
            
            # Print current joint positions
            joint_states = [p.getJointState(env.robot_id, i+1)[0] for i in range(6)]
            print(f"Current joint positions: {np.round(joint_states, 3)}")
            
            # Step simulation
            p.stepSimulation()
            time.sleep(0.01)
            
    finally:
        env.close()

def test_sinusoidal_velocity():
    """Test robot movement with sinusoidal velocity commands"""
    # Initialize environment
    env = XArmEnvironment(gui=True)
    
    # Amplitude for each joint
    amplitudes = np.array([0.5, 0.3, 0.3, 0.2, 0.2, 0.1])
    frequency = 0.5  # Hz
    
    try:
        print("Applying sinusoidal velocities")
        
        start_time = time.time()
        while time.time() - start_time < 10:  # Run for 10 seconds
            current_time = time.time() - start_time
            
            # Compute sinusoidal velocities
            velocities = amplitudes * np.sin(2 * np.pi * frequency * current_time)
            
            # Apply velocity commands
            for i in range(6):
                p.setJointMotorControl2(
                    bodyUniqueId=env.robot_id,
                    jointIndex=i+1,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=velocities[i],
                    force=100
                )
            
            # Print current joint positions
            joint_states = [p.getJointState(env.robot_id, i+1)[0] for i in range(6)]
            print(f"Current joint positions: {np.round(joint_states, 3)}")
            
            # Step simulation
            p.stepSimulation()
            time.sleep(0.01)
            
    finally:
        env.close()

if __name__ == "__main__":
    print("Testing constant velocity control...")
    test_constant_velocity()
    