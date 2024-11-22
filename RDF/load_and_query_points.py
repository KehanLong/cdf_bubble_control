import torch
import numpy as np
from panda_layer.panda_layer import PandaLayer
import bf_sdf
import trimesh
import trimesh.transformations as tra

class SDFQueryHelper:
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize robot and SDF model
        self.robot = PandaLayer(device, mesh_path="panda_layer/meshes/visual/*.stl")
        self.bp_sdf = bf_sdf.BPSDF(
            n_func=8,
            domain_min=-1.0,
            domain_max=1.0,
            robot=self.robot,
            device=device
        )
        self.model = torch.load('models/BP_8.pt')
        
    def generate_random_points(self, num_points=5):
        """Generate random points in the robot's workspace"""
        # Generate points in a reasonable workspace volume
        points = torch.rand(num_points, 3).to(self.device)
        # Scale points to reasonable workspace coordinates
        points = points * torch.tensor([1.0, 1.0, 0.5]).to(self.device)
        points = points + torch.tensor([0.3, -0.5, 0.2]).to(self.device)
        return points

    def query_sdf(self, points, robot_pose, joint_angles):
        """Query SDF values and gradients for given points and robot configuration"""
        sdf, joint_grad = self.bp_sdf.get_whole_body_sdf_with_joints_grad_batch(
            points, 
            robot_pose, 
            joint_angles, 
            self.model,
            used_links=[0,1,2,3,4,5,6,7,8]
        )
        return sdf.squeeze(), joint_grad.squeeze()

    def visualize_configuration(self, points, robot_pose, joint_angles, sdf_values):
        """Visualize robot and points using trimesh"""
        # Get robot meshes for current configuration
        robot_meshes = self.robot.get_forward_robot_mesh(robot_pose, joint_angles)[0]
        # Combine all robot meshes into one
        robot_mesh = np.sum(robot_meshes)
        
        # Create scene
        scene = trimesh.Scene()
        
        # Add robot mesh to scene
        robot_mesh.visual.face_colors = [150, 150, 200, 200]  # Light blue color
        scene.add_geometry(robot_mesh)

        # Add points as spheres
        points_np = points.cpu().numpy()
        for i, point in enumerate(points_np):
            # Create sphere at point location
            sphere = trimesh.primitives.Sphere(radius=0.02, center=point)
            
            # Color based on SDF value (red for negative/inside, blue for positive/outside)
            color = [255, 0, 0, 255] if sdf_values[i] < 0 else [0, 0, 255, 255]
            sphere.visual.face_colors = color
            scene.add_geometry(sphere)

        # Add coordinate frame
        scene.add_geometry(trimesh.creation.axis())
        
        # Show the scene
        scene.show()

if __name__ == "__main__":
    # Initialize controller
    controller = SDFQueryHelper()
    
    # Generate some random points
    test_points = controller.generate_random_points(num_points=3)
    
    # Create a base robot pose (identity transform)
    robot_pose = torch.eye(4).unsqueeze(0).to(controller.device)
    
    # Test different joint configurations
    joint_configs = [
        torch.zeros(1, 7).to(controller.device),  # Home position
        torch.ones(1, 7).to(controller.device) * 0.5,  # Some intermediate position
        torch.tensor([[0.3, 0.4, 0.0, -1.0, 0.0, 1.5, 0.0]]).to(controller.device)  # Custom position
    ]
    
    print("Testing SDF queries for different configurations:")
    print("-" * 50)
    
    for i, joints in enumerate(joint_configs):
        print(f"\nConfiguration {i+1}:")
        print(f"Joint angles: {joints.cpu().numpy()[0]}")
        
        sdf, gradients = controller.query_sdf(test_points, robot_pose, joints)
        
        print("\nPoint coordinates and their SDF values:")
        for j, (point, dist) in enumerate(zip(test_points, sdf)):
            print(f"Point {j+1}: {point.cpu().numpy()} -> Distance: {dist.item():.4f}")
            print(f"Gradient: {gradients[j].cpu().numpy()}")
        
        # Visualize this configuration
        controller.visualize_configuration(test_points, robot_pose, joints, sdf)