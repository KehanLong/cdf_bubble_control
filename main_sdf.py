import numpy as np
import jax.numpy as jnp
from data.dataset import SDFDataset
from utils.sdf_net import SDFNet
from training.train_sdf import train

def main():
    # Configuration
    num_links = 5
    hidden_size = 16
    num_layers = 4
    num_epochs = 500
    learning_rate = 0.002
    lambda_eikonal = 0.1
    threshold = 1e-4

    for i in range(num_links):
        # Load the saved dataset for each link
        link_data = np.load(f'train_dataset/link{i+1}_sdf_data.npy', allow_pickle=True).item()
        
        # Extract points and distances from the dataset
        points = link_data['points']
        distances = link_data['distances']
        
        # Create dataset
        dataset = SDFDataset(points, distances)
        
        # Create SDF network for each link
        net = SDFNet(hidden_size, num_layers)
        
        print(f"Training SDF network for Link {i+1}")
        # Train the model
        trained_params = train(net, dataset, num_epochs=num_epochs, learning_rate=learning_rate, lambda_eikonal=lambda_eikonal, threshold=threshold)
        
        # Save the trained model parameters for each link
        jnp.save(f"trained_models/link{i+1}_model_{num_layers}_{hidden_size}.npy", trained_params)
        print(f"Saved trained model for Link {i+1}")

if __name__ == "__main__":
    main()