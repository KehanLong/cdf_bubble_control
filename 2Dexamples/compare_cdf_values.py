import numpy as np
import torch
from utils.cdf_net import CDF_Net
from cdf_evaluate import load_learned_cdf
from tqdm import tqdm
from main_cdf import process_dataset


def compute_metrics(true_cdfs, predicted_cdfs):
    """Compute various error metrics."""
    # Mean Absolute Error
    mae = np.mean(np.abs(true_cdfs - predicted_cdfs))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((true_cdfs - predicted_cdfs)**2))
    
    # Maximum Absolute Error
    max_error = np.max(np.abs(true_cdfs - predicted_cdfs))
    
    # Metrics for truncated values (CDF < 0.2)
    truncated_mask = true_cdfs < 0.2
    truncated_true = true_cdfs[truncated_mask]
    truncated_pred = predicted_cdfs[truncated_mask]
    
    if len(truncated_true) > 0:
        truncated_mae = np.mean(np.abs(truncated_true - truncated_pred))
        truncated_rmse = np.sqrt(np.mean((truncated_true - truncated_pred)**2))
        truncated_max_error = np.max(np.abs(truncated_true - truncated_pred))
        truncated_count = len(truncated_true)
        truncated_percentage = (truncated_count / len(true_cdfs)) * 100
    else:
        truncated_mae = truncated_rmse = truncated_max_error = truncated_count = truncated_percentage = 0
    
    # Percentage of predictions within different thresholds
    within_001 = np.mean(np.abs(true_cdfs - predicted_cdfs) < 0.01) * 100
    within_005 = np.mean(np.abs(true_cdfs - predicted_cdfs) < 0.05) * 100
    within_01 = np.mean(np.abs(true_cdfs - predicted_cdfs) < 0.1) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Max Error': max_error,
        'Within 0.01': within_001,
        'Within 0.05': within_005,
        'Within 0.1': within_01,
        'Truncated MAE': truncated_mae,
        'Truncated RMSE': truncated_rmse,
        'Truncated Max Error': truncated_max_error,
        'Truncated Count': truncated_count,
        'Truncated Percentage': truncated_percentage
    }


def evaluate_cdf_pairs(model, configs, points, device, batch_size=1024):
    """Evaluate CDF values for corresponding config-point pairs."""
    model.eval()
    cdf_values = []
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(configs), batch_size):
            # Get batch
            batch_configs = configs[i:i+batch_size]
            batch_points = points[i:i+batch_size]
            
            # Encode configurations
            encoded_configs = np.concatenate([
                batch_configs, 
                np.sin(batch_configs), 
                np.cos(batch_configs)
            ], axis=-1)
            
            # Convert to tensors
            configs_tensor = torch.tensor(encoded_configs, dtype=torch.float32).to(device)
            points_tensor = torch.tensor(batch_points, dtype=torch.float32).to(device)
            
            # Concatenate for model input
            input_tensor = torch.cat([configs_tensor, points_tensor], dim=-1)
            
            # Get predictions
            batch_predictions = model(input_tensor)
            cdf_values.append(batch_predictions.cpu().numpy())
    
    return np.concatenate(cdf_values, axis=0)

def main():
    # Load validation dataset
    print("Loading validation dataset...")
    data = np.load('cdf_dataset/prepared_training_dataset/robot_cdf_validation_dataset_2_links.npz')
    configs = data['configurations']
    points = data['points']
    true_cdfs = data['cdf_values']

    configs, points, true_cdfs = process_dataset(configs, points, true_cdfs)
    
    # Load trained model
    print("\nLoading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_learned_cdf("trained_models/cdf_models/cdf_model_2_links_all.pt").to(device)
    
    # Get predictions
    print("\nComputing predictions...")
    predicted_cdfs = evaluate_cdf_pairs(model, configs, points, device).flatten()
    
    print(f"\nShapes:")
    print(f"  True CDFs: {true_cdfs.shape}")
    print(f"  Predicted CDFs: {predicted_cdfs.shape}")
    
    # Sample a few pairs for inspection
    print("\nSample predictions:")
    for i in range(5):
        idx = np.random.randint(len(true_cdfs))
        print(f"Sample {i+1}:")
        print(f"  Config: [{configs[idx][0]:.3f}, {configs[idx][1]:.3f}]")
        print(f"  Point: [{points[idx][0]:.3f}, {points[idx][1]:.3f}]")
        print(f"  True CDF: {true_cdfs[idx]:.4f}")
        print(f"  Predicted CDF: {predicted_cdfs[idx]:.4f}")
        print()
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(true_cdfs, predicted_cdfs)
    
    # Print results
    print("\nValidation Results:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 