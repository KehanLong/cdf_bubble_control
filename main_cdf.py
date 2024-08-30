import torch
import numpy as np
from data.dataset import RobotCDFDataset
from utils.cdf_net import CDFNet
from training.train_cdf import train

def preprocess_data(configurations, points, cdf_values):
    # Find indices where cdf_values are not NaN
    valid_indices = ~np.isnan(cdf_values)
    
    # Reshape valid_indices to match cdf_values shape
    valid_indices = valid_indices.reshape(cdf_values.shape)
    
    # Create masks for configurations and points
    config_mask = np.any(valid_indices, axis=1)
    point_mask = np.any(valid_indices, axis=0)
    
    # Filter data
    filtered_configs = configurations[config_mask]
    filtered_points = points[point_mask]
    filtered_cdf_values = cdf_values[config_mask][:, point_mask]
    
    return filtered_configs, filtered_points, filtered_cdf_values

def main():
    # Configuration
    input_size = 17  # 5 joints + 5 sin + 5 cos + 2 point coordinates
    mlp_layers = [256, 256, 256, 256, 256]
    num_epochs = 500
    learning_rate = 0.001
    batch_size = 256

    # Load the dataset
    data = np.load('cdf_dataset/robot_cdf_dataset.npy', allow_pickle=True).item()
    configurations = data['configurations']
    points = data['points']
    cdf_values = data['cdf_values']

    # Preprocess the data (remove NaN CDF values)
    configurations, points, cdf_values = preprocess_data(configurations, points, cdf_values)

    print("\nPreprocessed dataset info:")
    print(f"Configurations shape: {configurations.shape}")
    print(f"Points shape: {points.shape}")
    print(f"CDF values shape: {cdf_values.shape}")


    # Create dataset
    dataset = RobotCDFDataset(configurations, points, cdf_values)
    print(f"Dataset size: {len(dataset)}")


    # Create CDF network
    net = CDFNet(
        input_dims=input_size,
        output_dims=1,
        hidden_layers=mlp_layers,
    )


    print("Training CDF network")
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train(net, dataset, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, device=device)

    # Save the trained model parameters
    torch.save(trained_model.state_dict(), f"trained_models/cdf_models/cdf_model_{len(mlp_layers)}_{mlp_layers[0]}.pt")
    print("Saved trained CDF model")

if __name__ == "__main__":
    main()