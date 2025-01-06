import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import os
import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

from models.xarm_model import XArmFK
from robot_sdf import RobotSDF

def compute_robot_distances(robot_state, obstaclesX, robot_sdf, batch_size=50):
    """Compute distances between robot and obstacles using RobotSDF."""
    # For debugging: return constant distance with correct shape
    # if len(robot_state.shape) == 1:
    #     # Return shape [1, N] where N is number of points
    #     return torch.ones(1, 500, device=robot_state.device) * 0.5
    # else:
    #     # Return shape [B, N] where B is batch size and N is number of points
    #     return torch.ones(robot_state.shape[0], 500, device=robot_state.device) * 0.5

    # Add batch dimension if not present
    if len(robot_state.shape) == 1:
        robot_state = robot_state.unsqueeze(0)
    
    # Ensure obstacles are properly shaped [N, 3]
    if len(obstaclesX.shape) > 2:
        obstaclesX = obstaclesX.reshape(-1, 3)  # Flatten to [N, 3]
    
    total_states = robot_state.shape[0]
    device = robot_state.device
    all_sdf_values = []
    
    # Process robot states in batches
    for i in range(0, total_states, batch_size):
        batch_end = min(i + batch_size, total_states)
        batch_robot_state = robot_state[i:batch_end]
        
        # Query SDF values for this batch
        batch_obstacles = obstaclesX.unsqueeze(0).expand(batch_end - i, obstaclesX.shape[0], 3)
        
        # Query SDF values
        sdf_values = robot_sdf.query_sdf(
            batch_obstacles,  # [B, N, 3]
            batch_robot_state  # [B, 6]
        )
        
        all_sdf_values.append(sdf_values)
        torch.cuda.empty_cache()
    
    return torch.cat(all_sdf_values, dim=0)

def setup_mppi_controller(
    robot_sdf,
    robot_n=6,
    input_size=6,
    initial_horizon=10,
    samples=100,
    control_bound=3.0,
    dt=0.05,
    u_guess=None,
    use_GPU=True,
    costs_lambda=0.03,
    cost_goal_coeff=50.0,
    cost_safety_coeff=0.4,
    cost_perturbation_coeff=0.02,
    action_smoothing=0.5,
    noise_sigma=None
):
    device = 'cuda' if use_GPU and torch.cuda.is_available() else 'cpu'
    
    # Initialize noise parameters
    if noise_sigma is None:
        noise_sigma = 2 * torch.eye(input_size, device=device)
    noise_mu = torch.zeros(input_size, device=device)
    noise_dist = MultivariateNormal(noise_mu, covariance_matrix=noise_sigma)
    control_cov_inv = torch.inverse(noise_sigma)
    
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
        weights = weights.view(-1, 1, 1)
        
        # Add action smoothing while keeping the original weighting scheme
        weighted_perturbation = (perturbation * weights).sum(dim=0)
        new_U = U + weighted_perturbation / normalization_factor
        return (1.0 - action_smoothing) * U + action_smoothing * new_U
    
    def compute_rollout_costs(key, U, init_state, goal, obstaclesX, safety_margin, batch_size=10):
        """MPPI rollout with smaller batches"""
        torch.cuda.empty_cache()  # Clear at start
        
        # Sample noise using distribution
        perturbation = noise_dist.sample((samples, initial_horizon)).detach()
        perturbation = torch.clamp(perturbation, -1., 1.)
        
        # Add perturbation to nominal control sequence
        perturbed_control = U.unsqueeze(0) + perturbation
        perturbed_control = torch.clamp(perturbed_control, -control_bound, control_bound)
        perturbation = perturbed_control - U.unsqueeze(0)
        
        # Process samples in smaller batches
        all_costs = []
        
        for i in range(0, samples, batch_size):
            batch_end = min(i + batch_size, samples)
            batch_size_i = batch_end - i
            
            # Initialize state trajectory for this batch
            robot_states = torch.zeros((batch_size_i, robot_n, initial_horizon+1), device=device)
            robot_states[:, :, 0] = init_state.unsqueeze(0).expand(batch_size_i, -1)
            
            batch_costs = torch.zeros(batch_size_i, device=device)
            
            for t in range(initial_horizon):
                with torch.no_grad():  # Prevent gradient accumulation
                    current_state = robot_dynamics_step(
                        robot_states[:, :, t],
                        perturbed_control[i:batch_end, t]
                    )
                    robot_states[:, :, t+1] = current_state
                    
                    # Get end-effector positions for the batch
                    ee_positions = XArmFK().fkine(current_state)[:, -1]
                    ee_pos = ee_positions + torch.tensor([-0.6, 0.0, 0.625], device=device)
                    
                    # Goal cost
                    goal_dist = torch.norm(ee_pos - goal.unsqueeze(0), dim=1)
                    batch_costs += cost_goal_coeff * goal_dist
                    
                    # Safety cost with smaller batch size
                    distances = compute_robot_distances(current_state, obstaclesX, robot_sdf, batch_size=50)
                    min_distance = distances.min(dim=1)[0]
                    
                    batch_costs += cost_safety_coeff / torch.clamp(min_distance - safety_margin, min=0.01)
                    
                    # Control cost
                    ctrl_cost = ((perturbed_control[i:batch_end, t] - perturbation[i:batch_end, t]).unsqueeze(1) @ 
                                control_cov_inv @ 
                                perturbation[i:batch_end, t].unsqueeze(-1)).squeeze()
                    batch_costs += cost_perturbation_coeff * ctrl_cost
            
            all_costs.append(batch_costs.detach())  # Detach costs
            del robot_states  # Explicitly delete
            torch.cuda.empty_cache()
        
        # Combine results
        costs = torch.cat(all_costs)
        
        # Update nominal control sequence
        U_new = weighted_sum(U, perturbation, costs)
        
        # Compute final trajectory
        with torch.no_grad():
            states_final = torch.zeros((robot_n, initial_horizon+1), device=device)
            states_final[:, 0] = init_state
            for t in range(initial_horizon):
                states_final[:, t+1] = robot_dynamics_step(states_final[:, t], U_new[t])
        
        action = U_new[0].reshape(-1, 1)
        U_next = torch.cat([U_new[1:], U_new[-1:]], dim=0)
        
        # Clear all intermediate results
        del perturbation, perturbed_control, all_costs, costs
        torch.cuda.empty_cache()
        
        return states_final, action, U_next
    
    return compute_rollout_costs

def test_mppi():
    """Test MPPI controller with RobotSDF"""
    print("\n=== Starting MPPI Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store successful actions
    successful_actions = []
    
    try:
        # Initialize RobotSDF
        robot_sdf = RobotSDF(device)
        
        # Initial and goal states
        initial_state = torch.zeros(6, device=device)
        goal_state = torch.tensor([0.3961982, -0.3110882, 0.5596867], device=device, dtype=torch.float32) + torch.tensor([-0.6, 0.0, 0.625], device=device)
        
        print("\nInitial Configuration:")
        print(f"Initial joint state: {initial_state}")
        print(f"Goal state (world frame): {goal_state}")
        
        # Create test obstacles
        t = torch.linspace(0, 2*math.pi, 50, device=device)
        phi = torch.linspace(0, math.pi, 20, device=device)
        t, phi = torch.meshgrid(t, phi, indexing='ij')
        
        x = 0.3 + 0.2 * torch.sin(phi) * torch.cos(t)
        y = 0.0 + 0.2 * torch.sin(phi) * torch.sin(t)
        z = 0.5 + 0.2 * torch.cos(phi)
        
        obstacles = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        
        # Setup MPPI controller
        mppi = setup_mppi_controller(
            robot_sdf=robot_sdf,
            use_GPU=(device=='cuda'),
            samples=300,
            initial_horizon=10
        )
        
        # Initialize control sequence
        U = torch.zeros((10, 6), device=device)
        current_state = initial_state.clone()
        
        print("\nStarting MPPI optimization...")
        
        for epoch in range(100):
            torch.cuda.empty_cache()
            
            # Run one iteration of MPPI
            states_final, action, U = mppi(
                key=None,
                U=U,
                init_state=current_state,
                goal=goal_state,
                obstaclesX=obstacles,
                safety_margin=0.02
            )
            
            # Store the action
            successful_actions.append(action.cpu().numpy())
            
            # Update current state
            with torch.no_grad():
                current_state = current_state + action.squeeze() * 0.05
            
            # Compute current distance to goal
            current_ee_pos = XArmFK().fkine(current_state) + torch.tensor([-0.6, 0.0, 0.625], device=device)



            distance_to_goal = torch.norm(goal_state - current_ee_pos.squeeze()[-1, :])
            
            # Compute distance to obstacles
            distances = compute_robot_distances(current_state.unsqueeze(0), obstacles, robot_sdf)
            min_distance = distances.min().item()
            
            print(f"Epoch {epoch + 1}/20")
            print(f"Distance to goal: {distance_to_goal.item():.4f}")
            print(f"Distance to obstacle: {min_distance:.4f}")
            print(f"action: {action}")
            print("---")
            
            if distance_to_goal < 0.01:
                print("Reached goal!")
                # Save successful actions to file
                np.save('successful_mppi_actions.npy', np.array(successful_actions))
                break
            
            # Clear any remaining memory
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"MPPI test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mppi()