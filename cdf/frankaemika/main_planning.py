import pybullet as p
import pybullet_data
import numpy as np
import torch
import time
import os
from mlp import MLPRegression
from nn_cdf import CDF

class CDFVisualizer:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Load Franka robot
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Load CDF model
        self.cdf, self.model = self.load_cdf_model()
        
        # Create obstacles and get their point clouds
        self.obstacles = self.create_obstacles()
        self.obstacle_points = self.get_obstacle_points()
        
        # Store previous closest point for cleanup
        self.prev_closest_visual = None
        
        # Add these new parameters
        self.target_pos = np.array([0, 0.6, 0.5])  # Target position
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
    
    def create_obstacles(self):
        """Create various obstacles in the environment"""
        obstacles = []
        
        # Create a wall of small spheres
        for x in np.linspace(-0.3, 0.4, 10):
            for z in np.linspace(0.2, 0.5, 10):
                visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0.5, 0.5, 0.5, 0.7])
                body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, 
                                          basePosition=[x, 0.2, z])
                obstacles.append(body_id)
        
        # Create a ring of spheres
        # radius = 0.3
        # center = [0.5, -0.3, 0.5]
        # num_points = 16
        # for i in range(num_points):
        #     angle = 2 * np.pi * i / num_points
        #     x = center[0] + radius * np.cos(angle)
        #     y = center[1] + radius * np.sin(angle)
        #     visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0.5, 0.5, 0.5, 0.7])
        #     body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, 
        #                               basePosition=[x, y, center[2]])
        #     obstacles.append(body_id)
            
        return obstacles
    
    def get_obstacle_points(self):
        """Convert obstacles to point cloud"""
        points = []
        for obs in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(obs)
            points.append(pos)
        return torch.tensor(points, device=self.device)
    
    def query_cdf(self, robot_config):
        """Query CDF values for current robot configuration"""
        robot_config = torch.tensor(robot_config, device=self.device).reshape(1, 7)
        
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
        # Reset all obstacles to default color
        for obs in self.obstacles:
            p.changeVisualShape(obs, -1, rgbaColor=[0.5, 0.5, 0.5, 0.7])
        
        # Query all distances to find the closest point
        robot_config = torch.tensor([p.getJointState(self.robot_id, i)[0] for i in range(7)], 
                                  device=self.device).reshape(1, 7)
        
        with torch.no_grad():
            all_distances = self.cdf.inference(
                self.obstacle_points, 
                robot_config, 
                self.model
            ).reshape(-1)
        
        # Find the closest point
        closest_idx = torch.argmin(all_distances)
        closest_point = self.obstacle_points[closest_idx]
        
        # Highlight the closest obstacle in red
        p.changeVisualShape(self.obstacles[closest_idx], -1, rgbaColor=[1, 0, 0, 1])
        
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
        
        print(f"Minimum distance to obstacles: {min_dist:.3f}")
    
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
        
        # Use IK to get joint angles for target position
        target_orn = p.getQuaternionFromEuler([0, -np.pi, 0])  # Keep end effector pointing downward
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
            p.stepSimulation()
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

    def setup_joint_sliders(self):
        """Create sliders for controlling robot joints"""
        self.sliders = []
        for i in range(7):
            slider = p.addUserDebugParameter(f"Joint {i}", -3.14, 3.14, 0)
            self.sliders.append(slider)

def main():
    visualizer = CDFVisualizer()
    visualizer.setup_joint_sliders()
    visualizer.run()

if __name__ == "__main__":
    main()