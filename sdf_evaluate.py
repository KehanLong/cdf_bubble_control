
import jax
import numpy as np
import jax.numpy as jnp
from utils.sdf_net import SDFNet
from data.arm_2d_config import link_lengths, link_widths
import matplotlib.pyplot as plt
from utils.config import *

# jax.config.update('jax_platform_name', 'cpu')

@jax.jit
def evaluate_model(params, points):
    # Predict signed distances
    def apply_model(params, points):
        return SDFNet(HIDDEN_SIZE, 4).apply(params, points)
    
    outputs = apply_model(params, points)
    
    # Compute gradients
    grad_fn = jax.grad(lambda x: apply_model(params, x).sum())
    gradients = jax.vmap(grad_fn)(points)

    return outputs, gradients

def visualize_sdf_heatmap(link_length, link_width, points, distances, title):
    fig, ax = plt.subplots()
    # Reshape distances to match the grid shape
    n_points = int(jnp.sqrt(len(points)))
    distances = distances.reshape(n_points, n_points)
    
    # Create a heatmap of the signed distance field
    heatmap = ax.imshow(distances, cmap='coolwarm', extent=[-5, 5, -5, 5], origin='lower', vmin=-2, vmax=2)
    
    # Plot the zero level set
    contour = ax.contour(points[:, 0].reshape(n_points, n_points), points[:, 1].reshape(n_points, n_points), distances, levels=[0], colors='black', linewidths=1)
    
    # Plot the rectangle link
    rect = plt.Rectangle((-link_length/2, -link_width/2), link_length, link_width, edgecolor='b', facecolor='none', linewidth=1)
    ax.add_patch(rect)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Add a colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Signed Distance')
    
    plt.show()


def generate_workspace_points(n_points):
    x = np.linspace(-5, 5, n_points)
    y = np.linspace(-5, 5, n_points)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n_points * n_points)))
    return points

def main():
    # Load the trained models
    params_list = []

    trained_params = jnp.load(f"trained_models/tractor_model_4_16.npy", allow_pickle=True).item()
    params_list.append(trained_params)
    
    # Generate N points in the workspace
    n_points = 100
    points = generate_workspace_points(n_points)

    
    # Evaluate the models for each link
    link_length = link_lengths[0]
    link_width = link_widths[0]
    
    # Evaluate the SDF model for the link
    distances, gradients = evaluate_model(params_list[0], points)
    
    # Visualize the SDF heatmap
    title = f"Learned SDF for robot"
    visualize_sdf_heatmap(link_length, link_width, points, distances, title)

if __name__ == "__main__":
    main()