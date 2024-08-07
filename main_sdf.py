import numpy as np
import jax.numpy as jnp
from data.dataset import SDFDataset
from utils.sdf_net import SDFNet
from training.train_sdf import train



def main():

    # Load the saved dataset for each link
    link_data = np.load(f'train_dataset/link1_sdf_data.npy', allow_pickle=True).item()
    
    # Extract points and distances from the dataset
    points = link_data['points']
    distances = link_data['distances']
    
    # Create dataset
    dataset = SDFDataset(points, distances)
    
    # Create SDF network for each link
    hidden_size = 16
    num_layers = 4
    net = SDFNet(hidden_size, num_layers)
    
    # Train the model
    trained_params = train(net, dataset, num_epochs=500, learning_rate=0.003, lambda_eikonal=0.1, threshold=1e-4)
    
    # Save the trained model parameters for each link
    jnp.save(f"trained_models/trailer_model_4_16.npy", trained_params)

if __name__ == "__main__":
    main()