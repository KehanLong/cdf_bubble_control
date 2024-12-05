import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from mlp import MLPRegression
from nn_cdf import CDF
from operate_env_utils import FrankaEnvironment
import imageio


from cdf_bubble_planning import BubblePlanner

class CDFVisualizer:
    def __init__(self, target_pos, device='cuda', gui_set=True):
        self.device = device
        
        # Initialize environment using FrankaEnvironment with GUI disabled
        self.env = FrankaEnvironment(gui=gui_set, add_default_objects=True)
        self.robot_id = self.env.robot_id
        
        # Load CDF model
        self.cdf, self.model = self.load_cdf_model()
        
        # Get obstacle point cloud (now returns tuple of world and robot frame points)
        self.obstacle_points_world, self.obstacle_points = self.get_obstacle_points()
        
        # Store previous closest point for cleanup
        self.prev_closest_visual = None
        
        # Update target position to match goal config
        self.target_pos = target_pos  # Using the goal config you specified
        self.end_effector_index = 7
        
        # Create visual marker for target position
        self.create_target_marker()
        
        # For plotting CDF values over time
        self.cdf_values = []
        self.time_steps = []
        self.start_time = time.time()
        
        # Add bubble planner
        self.planner = BubblePlanner(self)
        
    def load_cdf_model(self):
        """Load the pretrained CDF model"""
        cdf = CDF(self.device)
        model = MLPRegression(
            input_dims=10,
            output_dims=1,
            mlp_layers=[1024, 512, 256, 128, 128],
            skips=[],
            act_fn=torch.nn.ReLU,
            nerf=True
        )
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, 'model_dict.pt')
        model.load_state_dict(torch.load(model_path)[49900])
        model.to(self.device)
        model.eval()
        
        return cdf, model
    
    def get_obstacle_points(self):
        """Get obstacle point cloud from environment"""
        points_world, points_robot = self.env.get_point_cloud(downsample=True, min_height=0.6)
        return (
            torch.tensor(points_world, device=self.device, dtype=torch.float32),
            torch.tensor(points_robot, device=self.device, dtype=torch.float32)
        )
    
    def query_cdf(self, robot_config):
        """Query CDF values for current robot configuration"""
        robot_config = torch.tensor(robot_config, device=self.device, dtype=torch.float32).reshape(1, 7)
        
        # Update point cloud every query to get latest obstacle positions
        _, self.obstacle_points = self.get_obstacle_points()
        
        with torch.no_grad():
            min_dist = self.cdf.inference_d_wrt_q(
                self.obstacle_points, 
                robot_config, 
                self.model,
                return_grad=False
            )
        
        # cdf model offset
        return min_dist.item() - 0.3
        #return 2.0
    
    def visualize_distances(self, min_dist):
        """Visualize the closest point"""
        # Query all distances to find the closest point
        robot_config = torch.tensor([p.getJointState(self.robot_id, i)[0] for i in range(7)], 
                                  device=self.device, dtype=torch.float32).reshape(1, 7)
        
        with torch.no_grad():
            all_distances = self.cdf.inference(
                self.obstacle_points, 
                robot_config, 
                self.model
            ).reshape(-1)
        
        # Find the closest point
        closest_idx = torch.argmin(all_distances)
        closest_point = self.obstacle_points[closest_idx]
        
        # Remove previous visual marker if it exists
        if self.prev_closest_visual is not None:
            p.removeUserDebugItem(self.prev_closest_visual)
        
        # Add debug text showing the distance
        text_pos = closest_point.cpu().numpy() + np.array([0, 0, 0.5])
        self.prev_closest_visual = p.addUserDebugText(
            f"cdf: {min_dist:.3f}",
            text_pos,
            textColorRGB=[1, 0, 0],
            textSize=2.
        )
    
    def create_target_marker(self):
        """Create a visual marker for the target position"""
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.8])
        self.target_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=self.target_pos
        )

    def get_end_effector_pos(self):
        """Get current end effector position"""
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(state[0])

    def move_to_target(self):
        """Move end effector to target using IK"""
        # Current end effector position
        current_pos = self.get_end_effector_pos()
        
        # Calculate error
        error = np.linalg.norm(self.target_pos - current_pos)
        
        target_orn = p.getQuaternionFromEuler([0, -np.pi, 0])
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            self.target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Apply joint positions
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                joint_poses[i],
                force=100
            )
        
        return error

    def execute_planned_path(self, planned_path, duration=5.0):
        """Execute planned path and record video"""
        if planned_path is None:
            print("No valid path to execute!")
            return

        planned_path = np.array(planned_path, dtype=np.float32)
        num_steps = len(planned_path)
        dt = duration / num_steps

        # Video recording setup
        width = 1920
        height = 1080
        frames = []

        # Initialize lists to store values
        self.cdf_values = []
        self.time_steps = []
        self.goal_distances = []
        start_time = time.time()
        
        # Create list for trajectory visualization
        ee_positions = []
        
        for i, config in enumerate(planned_path):
            # Set joint positions
            for joint_idx, joint_val in enumerate(config):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    float(joint_val),
                    force=100
                )
            
            # Step simulation
            for _ in range(10):
                p.stepSimulation()
            
            # Get and store end effector position
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
            ee_pos = ee_state[0]
            ee_positions.append(ee_pos)
            
            # Draw trajectory line
            if len(ee_positions) >= 2:
                p.addUserDebugLine(
                    ee_positions[-2],
                    ee_positions[-1],
                    lineColorRGB=[0, 0, 1],
                    lineWidth=2.0,
                    lifeTime=0
                )
            
            # Query CDF and record values
            min_dist = self.query_cdf(config)
            self.visualize_distances(min_dist)
            
            # Calculate distance to goal
            current_pos = self.get_end_effector_pos()
            goal_dist = np.linalg.norm(self.target_pos - current_pos)
            
            # Record values
            current_time = time.time() - start_time
            self.cdf_values.append(min_dist)
            self.time_steps.append(current_time)
            self.goal_distances.append(goal_dist)

            # Capture frame with rotating camera
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.0, 0.0, 1.0],
                distance=2.0,
                #yaw=(i / num_steps) * 100,  # Rotating camera
                yaw=0,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.1,
                farVal=100.0
            )
            
            # Get image
            _, _, rgb, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Convert to RGB and append to frames
            rgb_array = np.array(rgb)[:, :, :3]
            frames.append(rgb_array)
            
            # Step simulation
            self.env.step()
            time.sleep(dt)
        
        # Save video using imageio
        print("Saving video...")
        imageio.mimsave('robot_execution.mp4', frames, fps=int(1/dt))
        
        print("Motion complete!")
        self.plot_cdf_values()

    def run(self):
        """Main planning and execution loop"""
        print("Planning motion to target position...")
        
        # Get initial configuration (7 joints)
        initial_config = np.array([p.getJointState(self.robot_id, i)[0] for i in range(7)])
        print(f"Initial config: {initial_config}")
        initial_cdf = self.query_cdf(initial_config)
        print(f"Initial config CDF value: {initial_cdf}")
        
        # Calculate goal configuration using IK
        target_orn = p.getQuaternionFromEuler([0, -np.pi, 0])
        goal_config = np.array(p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            self.target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        ))[:7]
        #print(f"Goal config: {goal_config}")
        goal_cdf = self.query_cdf(goal_config)
        #print(f"Goal config CDF value: {goal_cdf}")
        
        # Check if goal configuration is safe
        if goal_cdf < 0.05:
            print("Warning: Goal configuration might be too close to obstacles!")
            print("Trying to find alternative IK solution...")
            
            # Try different orientations to find a safer goal configuration
            orientations = [
                p.getQuaternionFromEuler([0, -np.pi, 0]),
                p.getQuaternionFromEuler([0, -np.pi/2, 0]),
                p.getQuaternionFromEuler([0, -3*np.pi/4, 0]),
                p.getQuaternionFromEuler([0, -np.pi/4, 0])
            ]
            
            for orn in orientations:
                alt_goal_config = np.array(p.calculateInverseKinematics(
                    self.robot_id,
                    self.end_effector_index,
                    self.target_pos,
                    orn,
                    maxNumIterations=100,
                    residualThreshold=1e-5
                ))[:7]
                
                alt_cdf = self.query_cdf(alt_goal_config)
                print(f"Alternative goal config CDF value: {alt_cdf}")
                
                if alt_cdf >= 0.05:
                    goal_config = alt_goal_config
                    goal_cdf = alt_cdf
                    print("Found safer goal configuration!")
                    break
        
        try:
            # Use new bubble planner
            print("Planning path...")
            planned_path, bubbles = self.planner.plan(initial_config, goal_config)
            
            if planned_path is not None:
                print("Executing planned path...")
                self.execute_planned_path(planned_path)
                print("Motion complete!")
                self.plot_cdf_values()
            else:
                print("Failed to plan path!")
        except Exception as e:
            print(f"Planning failed with error: {str(e)}")

    def plot_cdf_values(self):
        """Plot CDF values and goal distances over time"""
        plt.figure(figsize=(10, 6))
        
        # Create two y-axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot CDF values on left y-axis
        line1 = ax1.plot(self.time_steps, self.cdf_values, 'b-', label='CDF Values', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=14)
        ax1.set_ylabel('CDF Value', color='b', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot goal distances on right y-axis
        line2 = ax2.plot(self.time_steps, self.goal_distances, 'r-', label='Distance to Goal', linewidth=2)
        ax2.set_ylabel('Distance to Goal Config', color='r', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=12)
        
        plt.title('CDF Values and Goal Distances During Execution', fontsize=16)
        plt.grid(True)
        plt.savefig('execution_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    target_pos = np.array([0.0, 0.3, 1.4])
    visualizer = CDFVisualizer(target_pos, gui_set=False)
    visualizer.run()

if __name__ == "__main__":
    main()
