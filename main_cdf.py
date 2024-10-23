import torch
import numpy as np
from data.dataset import RobotCDFDataset, RobotZeroDistanceDataset
from utils.cdf_net import CDF_Net
from training.train_cdf import train



def main():
    # Configuration
    data = np.load('cdf_dataset/robot_cdf_dataset_2_links.npy', allow_pickle=True).item()
    configurations = data['configurations']
    points = data['points']
    cdf_values = data['cdf_values']

    print("\nDetailed Dataset info:")
    print(f"Configurations shape: {np.array(configurations).shape}")
    print(f"Points shape: {np.array(points).shape}")
    print(f"CDF values shape: {np.array(cdf_values).shape}")
    print(f"Number of configurations: {len(configurations)}")
    print(f"Number of points: {len(points)}")

    
    num_links = configurations.shape[1]
    input_dims = num_links * 3 + 2  # Changed to account for raw angles, sin, cos, and 2D point
    output_dims = 1
    net= CDF_Net(input_dims=input_dims, hidden_dims=[512, 512, 512, 512], skip_in=[4], geometric_init=True).cuda()

    # Create dataset
    dataset = RobotCDFDataset(configurations, points, cdf_values)
    #dataset = RobotZeroDistanceDataset(configurations, points, cdf_values)
    print(f"Dataset size: {len(dataset)}")


    print("Training CDF network")
    # Train the model
    num_epochs = 10
    learning_rate = 0.0015
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train(net, dataset, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, device=device)

    # Save the trained model parameters
    torch.save(trained_model.state_dict(), f"trained_models/cdf_models/cdf_model_zeroconfigs_2_links.pt")
    print("Saved trained CDF model")

if __name__ == "__main__":
    main()
