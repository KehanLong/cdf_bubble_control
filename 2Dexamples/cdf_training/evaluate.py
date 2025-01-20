import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import os

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from robot_models import Robot2D
from cdf_computation import CDFDataProcessor, compute_cdf_and_gradients
from network import CDFNetwork

def create_evaluation_dataset(
    batch_q_size=200,
    batch_x_size=1000,
    device='cuda',
    save_path='data/evaluation_dataset_2d.pt',
    mini_batch_size=100
):
    """Create evaluation dataset by processing mini-batches of points and configurations"""

    src_dir = Path(__file__).parent
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"
    processor = CDFDataProcessor(contact_db_path, device=device)
    
    # Sample all points at once
    point_indices = torch.randint(0, len(processor.valid_points), (batch_x_size,))
    all_points = processor.valid_points[point_indices]
    
    # Sample all configurations at once
    all_configs = processor.sample_q(batch_q=batch_q_size)
    
    # Initialize final CDF values tensor
    final_cdf_values = torch.zeros(batch_x_size, batch_q_size)
    
    # Process in mini-batches
    for i in range(0, batch_x_size, mini_batch_size):
        for j in range(0, batch_q_size, mini_batch_size):
            x_start, x_end = i, min(i + mini_batch_size, batch_x_size)
            q_start, q_end = j, min(j + mini_batch_size, batch_q_size)
            
            start_time = time.time()
            print(f"Processing batch: points [{x_start}:{x_end}], configs [{q_start}:{q_end}]")
            
            # Get mini-batch data
            points_batch = all_points[x_start:x_end]
            configs_batch = all_configs[q_start:q_end]
            
            # Get corresponding contact data
            contact_configs_batch = [processor.contact_configs[idx] for idx in point_indices[x_start:x_end]]
            link_indices_batch = [processor.link_indices[idx] for idx in point_indices[x_start:x_end]]
            
            # Compute CDF values for this mini-batch
            cdf_values, _ = compute_cdf_and_gradients(
                points=points_batch,
                configs=configs_batch,
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
            
            # Store in final tensor
            final_cdf_values[x_start:x_end, q_start:q_end] = cdf_values.cpu()
            
            torch.cuda.empty_cache()
            print(f"Completed {(x_end-x_start)*(q_end-q_start)} values in {time.time() - start_time:.2f} seconds")
    
    # Save dataset
    torch.save({
        'points': all_points.cpu(),
        'configs': all_configs.cpu(),
        'gt_cdf_values': final_cdf_values,
    }, save_path)
    
    print(f"Evaluation dataset saved to {save_path}")
    print(f"Final shapes - Points: {all_points.shape}, Configs: {all_configs.shape}, "
          f"CDF Values: {final_cdf_values.shape}")
    
    return all_points, all_configs, final_cdf_values

def evaluate_quantitative(
    model_path,
    eval_dataset_path='data/evaluation_dataset_2d.pt',
    device='cuda'
):
    """Perform quantitative evaluation using a fixed evaluation dataset"""
    print("Loading evaluation dataset...")


    src_dir = Path(__file__).parent
    eval_dataset_path = src_dir /  "data" / "evaluation_dataset_2d.pt"
    data = torch.load(eval_dataset_path)
    points = data['points'].to(device)      # [N, 2]
    configs = data['configs'].to(device)    # [M, 2]
    gt_cdf_values = data['gt_cdf_values'].to(device)  # [N, M]
    
    print("Loading trained model...")
    model = CDFNetwork(input_dims=8, output_dims=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Computing model predictions...")
    with torch.no_grad():
        # Prepare inputs with positional encoding
        points_exp = points.unsqueeze(1).expand(-1, configs.shape[0], -1)  # [N, M, 2]
        configs_exp = configs.unsqueeze(0).expand(points.shape[0], -1, -1)  # [N, M, 2]
        
        # Apply positional encoding to configurations
        configs_sin = torch.sin(configs_exp)
        configs_cos = torch.cos(configs_exp)
        
        # Prepare network inputs
        inputs = torch.cat([
            configs_exp.reshape(-1, 2),
            configs_sin.reshape(-1, 2),
            configs_cos.reshape(-1, 2),
            points_exp.reshape(-1, 2)
        ], dim=1)
        
        # Get predictions
        pred_cdf_values = model(inputs).reshape(points.shape[0], configs.shape[0])
    
    # Compute metrics
    mse = torch.nn.functional.mse_loss(pred_cdf_values, gt_cdf_values).item()
    mae = torch.nn.functional.l1_loss(pred_cdf_values, gt_cdf_values).item()
    
    # Compute correlation coefficient
    pred_flat = pred_cdf_values.flatten()
    gt_flat = gt_cdf_values.flatten()
    correlation = torch.corrcoef(torch.stack([pred_flat, gt_flat]))[0,1].item()
    
    # Print results
    print("\nQuantitative Evaluation Results:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Correlation Coefficient: {correlation:.6f}")
    
    # Visualize predictions vs ground truth
    plt.figure(figsize=(10, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(gt_flat.detach().cpu(), pred_flat.detach().cpu(), alpha=0.1)
    plt.plot([0, gt_flat.max().item()], [0, gt_flat.max().item()], 'r--')
    plt.xlabel('Ground Truth CDF')
    plt.ylabel('Predicted CDF')
    plt.title('Predicted vs Ground Truth CDF')
    
    # Error histogram
    plt.subplot(122)
    errors = (pred_cdf_values - gt_cdf_values).detach().cpu().numpy()
    plt.hist(errors.flatten(), bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'points': points,
        'configs': configs,
        'predictions': pred_cdf_values,
        'ground_truth': gt_cdf_values
    }

def visualize_test_cases(model_path, device='cuda'):
    """Visualize specific test cases with the trained model"""
    # Initialize model and robot
    model = CDFNetwork(input_dims=8, output_dims=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    robot = Robot2D(device=device)
    
    # Define test configurations
    test_configs = [
        torch.tensor([0.0, 0.0], device=device),  # Home position
        torch.tensor([np.pi/4, np.pi/4], device=device),  # 45 degrees each
        torch.tensor([np.pi/2, -np.pi/2], device=device),  # Bent position
    ]
    
    # Create a grid of test points
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), device=device)
    
    # Plot for each configuration
    for i, config in enumerate(test_configs):
        plt.figure(figsize=(8, 8))
        
        # Prepare inputs with positional encoding
        config_exp = config.unsqueeze(0).expand(points.shape[0], -1)
        inputs = torch.cat([
            config_exp,
            torch.sin(config_exp),
            torch.cos(config_exp),
            points
        ], dim=1)
        
        # Get CDF predictions
        with torch.no_grad():
            cdf_values = model(inputs).cpu().numpy()
        
        # Plot CDF heatmap
        plt.scatter(X.flatten(), Y.flatten(), c=cdf_values, cmap='viridis')
        plt.colorbar(label='CDF Value')
        
        # Plot robot configuration
        joints = robot.forward_kinematics(config.unsqueeze(0))[0]
        joints = joints.cpu().numpy()
        plt.plot(joints[:, 0], joints[:, 1], 'r-', linewidth=2, label='Robot')
        plt.scatter(joints[:, 0], joints[:, 1], c='red', s=50)
        
        plt.title(f'Configuration {i+1}: [{config[0]:.2f}, {config[1]:.2f}]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

def compare_computation_efficiency(model_path, device='cuda', num_trials=100):
    """Compare computation time between learned model and template-based approach"""
    print("\nComparing computation efficiency...")
    
    # Initialize models and processor
    model = CDFNetwork(input_dims=8, output_dims=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    src_dir = Path(__file__).parent
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"
    processor = CDFDataProcessor(contact_db_path, device=device)
    
    # Test cases: (num_points, num_configs)
    test_cases = [
        (1, 1),
        (1, 10),
        (1, 100),
        (10, 1),
    ]
    
    results = {}
    
    for num_points, num_configs in test_cases:
        key = f"points_{num_points}_configs_{num_configs}"
        results[key] = {'learned': [], 'template': []}
        
        print(f"\nTest case: {num_points} points, {num_configs} configs")
        
        for _ in range(num_trials):
            # Sample points and configs
            point_indices = torch.randint(0, len(processor.valid_points), (num_points,))
            points = processor.valid_points[point_indices]
            configs = processor.sample_q(batch_q=num_configs)
            
            # Get contact data and ensure it's on the correct device
            contact_configs_batch = []
            link_indices_batch = []
            for idx in point_indices:
                # Reshape contact configs to 2D tensor [N, 2]
                contact_configs = torch.tensor(processor.contact_configs[idx], device=device)
                if contact_configs.dim() == 1:
                    contact_configs = contact_configs.reshape(-1, 2)
                contact_configs_batch.append(contact_configs)
                
                # Convert link indices to tensor
                link_indices = torch.tensor(processor.link_indices[idx], device=device)
                link_indices_batch.append(link_indices)
            
            # Time learned model
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # Prepare inputs with positional encoding
                points_exp = points.unsqueeze(1).expand(-1, configs.shape[0], -1)
                configs_exp = configs.unsqueeze(0).expand(points.shape[0], -1, -1)
                configs_sin = torch.sin(configs_exp)
                configs_cos = torch.cos(configs_exp)
                
                inputs = torch.cat([
                    configs_exp.reshape(-1, 2),
                    configs_sin.reshape(-1, 2),
                    configs_cos.reshape(-1, 2),
                    points_exp.reshape(-1, 2)
                ], dim=1)
                
                _ = model(inputs)
            
            torch.cuda.synchronize()
            learned_time = time.time() - start_time
            results[key]['learned'].append(learned_time)
            
            # Time template-based approach
            torch.cuda.synchronize()
            start_time = time.time()
            
            from cdf_computation import compute_cdf_and_gradients
            _, _ = compute_cdf_and_gradients(
                points=points,
                configs=configs,
                contact_configs=contact_configs_batch,
                link_indices=link_indices_batch,
                device=device
            )
            
            torch.cuda.synchronize()
            template_time = time.time() - start_time
            results[key]['template'].append(template_time)
        
        # Compute statistics
        learned_mean = np.mean(results[key]['learned'])
        learned_std = np.std(results[key]['learned'])
        template_mean = np.mean(results[key]['template'])
        template_std = np.std(results[key]['template'])
        speedup = template_mean / learned_mean
        
        print(f"Learned model:   {learned_mean*1000:.2f} ± {learned_std*1000:.2f} ms")
        print(f"Template-based:  {template_mean*1000:.2f} ± {template_std*1000:.2f} ms")
        print(f"Speedup factor:  {speedup:.1f}x")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(test_cases))
    width = 0.35
    
    learned_means = [np.mean(results[f"points_{p}_configs_{c}"]['learned']) * 1000 
                    for p, c in test_cases]
    template_means = [np.mean(results[f"points_{p}_configs_{c}"]['template']) * 1000 
                     for p, c in test_cases]
    
    learned_stds = [np.std(results[f"points_{p}_configs_{c}"]['learned']) * 1000 
                   for p, c in test_cases]
    template_stds = [np.std(results[f"points_{p}_configs_{c}"]['template']) * 1000 
                    for p, c in test_cases]
    
    plt.bar(x - width/2, learned_means, width, label='Learned Model',
            yerr=learned_stds, capsize=5)
    plt.bar(x + width/2, template_means, width, label='Template-based',
            yerr=template_stds, capsize=5)
    
    plt.xlabel('Test Cases')
    plt.ylabel('Computation Time (ms)')
    plt.title('Computation Time Comparison')
    plt.xticks(x, [f'{p}p,{c}c' for p, c in test_cases], rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    src_dir = project_root
    
    # Create evaluation dataset (only need to run once)
    save_path = src_dir / 'cdf_training' / 'data' / 'evaluation_dataset_2d.pt'
    # create_evaluation_dataset(batch_q_size=200, batch_x_size=400, device='cuda', save_path=save_path)
    
    # Run evaluation
    model_path = src_dir / 'trained_models' / 'cdf' / 'best_model_relu.pth'
    
    # Run efficiency comparison
    # efficiency_results = compare_computation_efficiency(model_path=model_path)
    
    # Run other evaluations
    metrics = evaluate_quantitative(model_path=model_path)
    visualize_test_cases(model_path=model_path) 