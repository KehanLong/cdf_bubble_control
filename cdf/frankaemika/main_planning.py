import pybullet as p
import pybullet_data
import numpy as np
import torch
import time
import os
from mlp import MLPRegression
from nn_cdf import CDF
from operate_env_utils import FrankaEnvironment

class CDFVisualizer:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize environment using FrankaEnvironment
        self.env = FrankaEnvironment(gui=True)
        self.robot_id = self.env.robot_id
        
        # Load CDF model
        self.cdf, self.model = self.load_cdf_model()
        
        # Get obstacle point cloud
        self.obstacle_points = self.get_obstacle_points()
        
        # Store previous closest point for cleanup
        self.prev_closest_visual = None
        
        # Add these new parameters
        self.target_pos = np.array([0.4, 0.5, 0.6])  # Target position
        self.end_effector_index = 7  # Franka's end effector link index
        
        # Create visual marker for target position
        self.create_target_marker()
        
        # For plotting CDF values over time
        self.cdf_values = []
        self.time_steps = []
        self.start_time = time.time()
        
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
        points = self.env.get_point_cloud(downsample=True, min_height=0.6)
        return torch.tensor(points, device=self.device, dtype=torch.float32)
    
    def query_cdf(self, robot_config):
        """Query CDF values for current robot configuration"""
        robot_config = torch.tensor(robot_config, device=self.device, dtype=torch.float32).reshape(1, 7)

        print('obstacle points', self.obstacle_points.shape)
        
        with torch.no_grad():
            min_dist = self.cdf.inference_d_wrt_q(
                self.obstacle_points, 
                robot_config, 
                self.model,
                return_grad=False
            )
        
        return min_dist.item()
    
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
        text_pos = closest_point.cpu().numpy() + np.array([0, 0, 0.05])
        self.prev_closest_visual = p.addUserDebugText(
            f"cdf: {min_dist:.3f}",
            text_pos,
            textColorRGB=[1, 0, 0],
            textSize=1.5
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

    def run(self):
        """Main visualization loop"""
        print("Moving to target position...")
        
        while True:
            # Move robot toward target
            error = self.move_to_target()
            
            # Get current robot joint states
            joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
            
            # Query CDF and visualize
            min_dist = self.query_cdf(joint_states)
            self.visualize_distances(min_dist)
            
            # Record CDF value and time
            current_time = time.time() - self.start_time
            self.cdf_values.append(min_dist)
            self.time_steps.append(current_time)
            
            # Print current status
            current_pos = self.get_end_effector_pos()
            print(f"Current position: {current_pos}, Error: {error:.3f}, Min distance: {min_dist:.3f}")
            
            # Step simulation
            self.env.step()
            time.sleep(1./240.)
            
            # Optional: Stop if we're close enough to target
            if error < 0.01:
                print("Reached target position!")
                self.plot_cdf_values()
                break

    def plot_cdf_values(self):
        """Plot CDF values over time"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.plot(self.time_steps, self.cdf_values, 'b-', label='CDF Value')
            plt.xlabel('Time (s)')
            plt.ylabel('Minimum Distance (m)')
            plt.title('CDF Values During Motion')
            plt.grid(True)
            plt.legend()
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")

def main():
    visualizer = CDFVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()