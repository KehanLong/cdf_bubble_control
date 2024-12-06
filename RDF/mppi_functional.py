import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import numpy as np
import os
from panda_layer.panda_layer import PandaLayer
from bf_sdf import BPSDF

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_transformation_matrix(alpha, a, d, theta):
    """
    Compute DH transformation matrix.
    Args:
        alpha, a, d: scalar DH parameters
        theta: batched joint angles [batch_size]
    Returns:
        T: batched transformation matrices [batch_size, 4, 4]
    """
    batch_size = theta.shape[0]
    device = theta.device
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Create batch of transformation matrices
    T = torch.zeros(batch_size, 4, 4, device=device)
    
    # Fill in the transformation matrix
    T[:, 0, 0] = cos_theta
    T[:, 0, 1] = -sin_theta
    T[:, 0, 2] = 0
    T[:, 0, 3] = a
    
    T[:, 1, 0] = sin_theta * torch.cos(alpha)
    T[:, 1, 1] = cos_theta * torch.cos(alpha)
    T[:, 1, 2] = -torch.sin(alpha)
    T[:, 1, 3] = -torch.sin(alpha) * d
    
    T[:, 2, 0] = sin_theta * torch.sin(alpha)
    T[:, 2, 1] = cos_theta * torch.sin(alpha)
    T[:, 2, 2] = torch.cos(alpha)
    T[:, 2, 3] = torch.cos(alpha) * d
    
    T[:, 3, 0] = 0
    T[:, 3, 1] = 0
    T[:, 3, 2] = 0
    T[:, 3, 3] = 1
    
    return T

def forward_kinematics_batch(joint_angles):
    """
    Compute forward kinematics for batched joint angles.
    Args:
        joint_angles: batched joint angles [batch_size, 7]
    Returns:
        end_effector_positions: batched positions [batch_size, 3]
    """
    batch_size = joint_angles.shape[0]
    device = joint_angles.device
    
    # DH parameters for Franka Emika Panda
    dh_params = torch.tensor([
        [0,      0,        0.333,   0],
        [-math.pi/2, 0,    0,       0],
        [math.pi/2,  0,    0.316,   0],
        [math.pi/2,  0.0825, 0,     0],
        [-math.pi/2, -0.0825, 0.384, 0],
        [math.pi/2,  0,      0,      0],
        [math.pi/2,  0.088,  0.107,  0]
    ], device=device)
    
    # Initialize transformation matrix
    T = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Compute forward kinematics for each joint
    for i in range(7):
        alpha, a, d = dh_params[i, 0:3]
        theta = joint_angles[:, i]
        
        # Compute transformation matrix for current joint
        Ti = get_transformation_matrix(alpha, a, d, theta)
        
        # Update cumulative transformation
        T = torch.bmm(T, Ti)
    
    # Extract end effector position
    end_effector_positions = T[:, 0:3, 3]
    
    return end_effector_positions

def compute_robot_distances(robot_state, obstaclesX, learned_CDF, batch_size=50):
    """Compute distances between robot and obstacles using learned CDF with batching."""
    # Add batch dimension if not present
    if len(robot_state.shape) == 1:
        robot_state = robot_state.unsqueeze(0)
    
    total_states = robot_state.shape[0]
    device = robot_state.device
    all_sdf_values = []
    
    # Process robot states in batches
    for i in range(0, total_states, batch_size):
        batch_end = min(i + batch_size, total_states)
        batch_robot_state = robot_state[i:batch_end]
        
        robot_pose = torch.eye(4, device=device).unsqueeze(0).expand(batch_end - i, -1, -1)
        
        # Get SDF values for this batch
        sdf_values, _ = learned_CDF.get_whole_body_sdf_batch(
            obstaclesX,
            robot_pose,
            batch_robot_state,
            learned_CDF.model,
            use_derivative=False
        )
        
        all_sdf_values.append(sdf_values)
        torch.cuda.empty_cache()  # Clear cache after each batch
    
    return torch.cat(all_sdf_values, dim=0)

def setup_mppi_controller(
    learned_CDF,
    robot_n=7,
    input_size=7,
    initial_horizon=10,
    samples=500,
    control_bound=2.0,
    dt=0.05,
    u_guess=None,
    use_GPU=True,
    costs_lambda=0.03,
    cost_goal_coeff=15.0,
    cost_safety_coeff=0.8,
    cost_perturbation_coeff=0.02,
    cost_goal_coeff_final=12.0,
    cost_safety_coeff_final=1.0
):
    device = 'cuda' if use_GPU and torch.cuda.is_available() else 'cpu'
    
    control_mu = torch.zeros(input_size, device=device)
    control_cov = 2 * torch.eye(input_size, device=device)
    control_cov_inv = torch.inverse(control_cov)
    
    if u_guess is not None:
        U = u_guess
    else:
        U = 0.0 * torch.ones((initial_horizon, input_size), device=device)
    
    def robot_dynamics_step(state, input_):
        return state + input_ * dt
    
    def weighted_sum(U, perturbation, costs):
        costs = costs - costs.min()
        costs = costs / costs.max()
        weights = torch.exp(-1.0/costs_lambda * costs)
        normalization_factor = weights.sum()
        weights = weights.view(-1, 1, 1)  # Reshape to [samples, 1, 1] for proper broadcasting
        weighted_perturbation = (perturbation * weights).sum(dim=0)
        return U + weighted_perturbation / normalization_factor
    
    def compute_rollout_costs(key, U, init_state, goal, obstaclesX, safety_margin, batch_size=50):
        """MPPI rollout with batching"""
        # Sample noise
        dist = MultivariateNormal(control_mu, control_cov)
        perturbation = dist.sample((samples, initial_horizon))
        perturbation = torch.clamp(perturbation, -0.4, 0.4)
        
        # Add perturbation to nominal control sequence
        perturbed_control = U.unsqueeze(0) + perturbation
        perturbed_control = torch.clamp(perturbed_control, -control_bound, control_bound)
        perturbation = perturbed_control - U.unsqueeze(0)
        
        # Process samples in batches
        all_costs = []
        all_states = []
        
        for i in range(0, samples, batch_size):
            batch_end = min(i + batch_size, samples)
            batch_size_i = batch_end - i
            
            # Initialize state trajectory for this batch
            robot_states = torch.zeros((batch_size_i, robot_n, initial_horizon+1), device=device)
            robot_states[:, :, 0] = init_state.unsqueeze(0).expand(batch_size_i, -1)
            
            batch_costs = torch.zeros(batch_size_i, device=device)
            
            # Rollout trajectories for this batch
            for t in range(initial_horizon):
                # Update states
                robot_states[:, :, t+1] = robot_dynamics_step(
                    robot_states[:, :, t],
                    perturbed_control[i:batch_end, t]
                )
                
                current_state = robot_states[:, :, t+1]
                ee_pos = forward_kinematics_batch(current_state) + torch.tensor([-0.6, 0.1, 0.625], device='cuda')
                
                # Goal cost
                goal_dist = torch.norm(ee_pos - goal, dim=1)
                batch_costs += cost_goal_coeff * goal_dist
                
                # Safety cost
                distances = compute_robot_distances(current_state, obstaclesX, learned_CDF, batch_size=batch_size)
                min_distance = distances.min(dim=1)[0]
                batch_costs += cost_safety_coeff / torch.clamp(min_distance - safety_margin, min=0.01)
                
                # Control cost
                ctrl_cost = ((perturbed_control[i:batch_end, t] - perturbation[i:batch_end, t]).unsqueeze(1) @ 
                            control_cov_inv @ 
                            perturbation[i:batch_end, t].unsqueeze(-1)).squeeze()
                batch_costs += cost_perturbation_coeff * ctrl_cost
            
            all_costs.append(batch_costs)
            all_states.append(robot_states)
            torch.cuda.empty_cache()  # Clear cache after each batch
        
        # Combine results
        costs = torch.cat(all_costs)
        robot_states = torch.cat(all_states)
        
        # Update nominal control sequence
        U = weighted_sum(U, perturbation, costs)
        
        # Compute final trajectory
        states_final = torch.zeros((robot_n, initial_horizon+1), device=device)
        states_final[:, 0] = init_state
        for t in range(initial_horizon):
            states_final[:, t+1] = robot_dynamics_step(states_final[:, t], U[t])
        
        action = U[0].reshape(-1, 1)
        U = torch.cat([U[1:], U[-1:]], dim=0)
        
        # Fix the reshape dimensions
        sampled_states = robot_states[:, :, :-1]  # Remove the last state
        sampled_states = sampled_states.transpose(1, 2).reshape(-1, robot_n)  # Reshape to [samples*horizon, robot_n]
        
        return sampled_states, states_final, action, U
    
    return compute_rollout_costs

def test_mppi():
    """Test MPPI controller with pre-trained model"""
    print("Testing MPPI controller...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Initialize robot and SDF model
        robot = PandaLayer(device, mesh_path="panda_layer/meshes/visual/*.stl")
        bp_sdf = BPSDF(
            n_func=8,
            domain_min=-1.0,
            domain_max=1.0,
            robot=robot,
            device=device
        )
        
        # Load pre-trained model
        model_path = os.path.join(CUR_DIR, 'models', 'BP_8.pt')
        bp_sdf.model = torch.load(model_path)
        
        # Initial and goal states
        initial_state = torch.zeros(7, device=device)
        goal_state = torch.tensor([0.2, 0.3, 0.8], device=device)  # Task space goal

        # Get initial distance to goal
        initial_ee_pos = forward_kinematics_batch(initial_state.unsqueeze(0))
        print('Initial distance to goal: ', torch.norm(goal_state - initial_ee_pos.squeeze()))
        
        # Create test obstacles (sphere centered at [0.3, 0, 0.5])
        t = torch.linspace(0, 2*math.pi, 50, device=device)
        phi = torch.linspace(0, math.pi, 20, device=device)
        t, phi = torch.meshgrid(t, phi, indexing='ij')
        
        x = 0.3 + 0.2 * torch.sin(phi) * torch.cos(t)
        y = 0.0 + 0.2 * torch.sin(phi) * torch.sin(t)
        z = 0.5 + 0.2 * torch.cos(phi)
        
        obstacles = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        
        # Setup MPPI controller
        mppi = setup_mppi_controller(
            learned_CDF=bp_sdf,
            use_GPU=(device=='cuda'),
            samples=500,
            initial_horizon=10
        )
        
        # Initialize control sequence
        U = torch.zeros((10, 7), device=device)
        current_state = initial_state.clone()
        
        # Run MPPI for multiple epochs
        n_epochs = 20
        print("\nStarting MPPI optimization...")
        
        for epoch in range(n_epochs):
            # Run one iteration of MPPI
            sampled_states, states_final, action, U = mppi(
                key=None,
                U=U,
                init_state=current_state,
                goal=goal_state,
                obstaclesX=obstacles,
                safety_margin=0.05,
                batch_size=50
            )
            
            # Update current state
            current_state = current_state + action.squeeze() * 0.05  # dt = 0.05
            
            # Compute and print current distance to goal
            current_ee_pos = forward_kinematics_batch(current_state.unsqueeze(0))
            distance_to_goal = torch.norm(goal_state - current_ee_pos.squeeze())

            # Add batch dimension to current_state for compute_robot_distances
            distances = compute_robot_distances(current_state.unsqueeze(0), obstacles, bp_sdf, batch_size=50)
            
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Distance to goal: {distance_to_goal.item():.4f}")
            print(f"Distance to obstacle: {distances.min().item():.4f}")
            print("---")
            
            # Optional: Early stopping if we're close enough to goal
            if distance_to_goal < 0.01:
                print("Reached goal!")
                break
                
            # Clear some memory
            torch.cuda.empty_cache()
        
        print("\nMPPI optimization completed!")
        print(f"Final distance to goal: {distance_to_goal.item():.4f}")
        print(f"Final EE position: {current_ee_pos.squeeze().cpu().numpy()}")
        print(f"Final joint state: {current_state.cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"MPPI test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mppi()
    

    
        
