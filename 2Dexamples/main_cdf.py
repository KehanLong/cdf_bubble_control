import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import RobotCDFDataset
from utils.cdf_net import CDF_Net
from training.train_cdf import train

def process_dataset(configurations, points, cdf_values):
    """
    Process the dataset to expand it for training if needed.
    
    Args:
    configurations (np.array): Shape (M, num_links)
    points (np.array): Shape (N, 2)
    cdf_values (np.array): Shape (M, N) - Matrix of CDF values
    
    Returns:
    tuple: (configs, points, cdf_values) in expanded format
    """
    # Check if configurations and points have the same first dimension
    if configurations.shape[0] == points.shape[0]:
        print("Dataset is already in correct format, skipping processing")
        return configurations, points, cdf_values
        
    # If not expanded, process the dataset
    M = configurations.shape[0]
    N = points.shape[0]
    
    print("Processing dataset to expanded format...")
    # Expand configurations
    expanded_configs = np.repeat(configurations, N, axis=0)
    # Expand points
    expanded_points = np.tile(points, (M, 1))
    # Flatten cdf_values matrix
    expanded_cdf_values = cdf_values.flatten()
    
    print(f"Processed Dataset info:")
    print(f"  Expanded configurations shape: {expanded_configs.shape}")
    print(f"  Expanded points shape: {expanded_points.shape}")
    print(f"  Expanded CDF values shape: {expanded_cdf_values.shape}")
    
    return expanded_configs, expanded_points, expanded_cdf_values

def load_dataset(filename):
    data = np.load(filename)
    configurations = data['configurations']
    points = data['points']
    cdf_values = data['cdf_values']
    
    print("Loaded Dataset info:")
    print(f"Configurations shape: {configurations.shape}")
    print(f"Points shape: {points.shape}")
    print(f"CDF values shape: {cdf_values.shape}")
    
    return configurations, points, cdf_values

def load_pretrained_model(model, model_path):
    """
    Load a pretrained model if it exists.
    
    Args:
    model: The model instance to load weights into
    model_path: Path to the pretrained model weights
    
    Returns:
    bool: True if model was loaded, False otherwise
    """
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")
        return True
    except FileNotFoundError:
        print(f"No pretrained model found at {model_path}")
        return False

def main():
    # Load the dataset
    configurations, points, cdf_values = load_dataset('cdf_dataset/prepared_training_dataset/robot_cdf_truncated_dataset_new.npz')

    # Reshape configurations for 1-link case if needed
    if len(configurations.shape) == 1:
        configurations = configurations.reshape(-1, 1)

    print("Original Dataset info:")
    print(f"Configurations shape: {configurations.shape}")
    print(f"Points shape: {points.shape}")
    print(f"CDF values shape: {cdf_values.shape}")

    # Process the dataset
    expanded_configs, expanded_points, expanded_cdf_values = process_dataset(configurations, points, cdf_values)

    # Create the dataset for training
    dataset = RobotCDFDataset(expanded_configs, expanded_points, expanded_cdf_values)

    print(f"Dataset size: {len(dataset)}")
    print("Training CDF network")

    # Set up the network
    num_links = configurations.shape[1]
    input_dims = num_links * 3 + 2  # Raw angles, sin, cos, and 2D point
    output_dims = 1  # 1 CDF value   
    net = CDF_Net(
        input_dims=input_dims, 
        output_dims=output_dims,  # Now outputs per-link distances
    ).cuda()

    # Try to load pretrained model
    model_path = "trained_models/cdf_models/cdf_model_2_links_truncated_new.pt"
    loaded_pretrained = load_pretrained_model(net, model_path)

    loaded_pretrained = False
    
    if not loaded_pretrained:
        print("Starting training from scratch")
    else:
        print("Continuing training from pretrained model")
    
    
    # Train the model
    num_epochs = 100
    learning_rate = 1.5e-4
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # print("\nDebug: First batch from dataset:")
    # sample_batch = next(iter(dataloader))
    # configs_batch, cdfs_batch = sample_batch
    # print(f"Batch shapes:")
    # print(f"  Configurations: {configs_batch.shape}")
    # print(f"  CDF values: {cdfs_batch.shape}")
    # print("\nFirst item in batch:")
    # print(f"  Configuration: {configs_batch[0]}")
    # print(f"  CDF value: {cdfs_batch[0]}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, best_loss = train(net, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, device=device, 
                                     use_improved_sampler=False, loss_threshold=1e-4, training_mode='truncated')

    print(f"Best loss: {best_loss}")
    torch.save(trained_model.state_dict(), f"trained_models/cdf_models/cdf_model_2_links_truncated_new.pt")
    print("Saved trained CDF model")

if __name__ == "__main__":
    main()
