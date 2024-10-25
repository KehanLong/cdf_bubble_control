import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import RobotCDFDataset, RobotZeroDistanceDataset
from utils.cdf_net import CDF_Net
from training.train_cdf import train

def process_dataset(configurations, points, cdf_values):
    """
    Process the dataset to expand it for training.
    
    Args:
    configurations (np.array): Shape (M, num_links)
    points (np.array): Shape (N, 2)
    cdf_values (np.array): Shape (M, N)
    
    Returns:
    tuple: (expanded_configs, expanded_points, expanded_cdf_values)
    """
    M, num_links = configurations.shape
    N = points.shape[0]
    
    # Expand configurations
    expanded_configs = np.repeat(configurations, N, axis=0)
    
    # Expand points
    expanded_points = np.tile(points, (M, 1))
    
    # Flatten cdf_values
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

def main():
    # Load the dataset
    configurations, points, cdf_values = load_dataset('cdf_dataset/robot_cdf_dataset_2_links_new.npz')

    print("Original Dataset info:")
    print(f"Configurations shape: {configurations.shape}")
    print(f"Points shape: {points.shape}")
    print(f"CDF values shape: {cdf_values.shape}")

    # Process the dataset
    expanded_configs, expanded_points, expanded_cdf_values = process_dataset(configurations, points, cdf_values)

    # Create the dataset for training
    dataset = RobotCDFDataset(expanded_configs, expanded_points, expanded_cdf_values)

    # Set up the network
    num_links = configurations.shape[1]
    input_dims = num_links * 3 + 2  # Raw angles, sin, cos, and 2D point
    net = CDF_Net(input_dims=input_dims, hidden_dims=[512, 512, 512, 512], skip_in=[4], geometric_init=True).cuda()

    #dataset = RobotZeroDistanceDataset(configurations, points, cdf_values)
    print(f"Dataset size: {len(dataset)}")

    print("Training CDF network")
    # Train the model
    num_epochs = 800
    learning_rate = 0.0015
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, best_loss = train(net, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, device=device, use_improved_sampler=False, loss_threshold=0.003)

    # Save the trained model parameters
    torch.save(trained_model.state_dict(), f"trained_models/cdf_models/cdf_model_2_links.pt")
    print("Saved trained CDF model")

if __name__ == "__main__":
    main()
