import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch
from flax import linen as nn
from flax.core.frozen_dict import freeze
import time

from data.arm_2d_config import NUM_LINKS, shapes


from utils.cdf_config import *
from utils.cdf_net import CDFNet, CDFNet_JAX
from arm_2d_utils import forward_kinematics, transform_shape


def load_learned_cdf(trained_model_path="trained_models/cdf_models/cdf_model_5_256.pt"):
    # Load the trained PyTorch model
    pytorch_net = CDFNet(input_dims=INPUT_SIZE, output_dims=OUTPUT_SIZE, hidden_layers=[HIDDEN_SIZE] * (NUM_LAYERS - 1))
    
    # Load the state dict
    state_dict = torch.load(trained_model_path)
    
    # Load the state dict directly into the model
    pytorch_net.load_state_dict(state_dict)
    pytorch_net.eval()  # Set the model to evaluation mode

    # Define your JAX model
    jax_net = CDFNet_JAX(hidden_layers=(HIDDEN_SIZE,) * (NUM_LAYERS - 1), output_dims=OUTPUT_SIZE)

    # Transfer weights from PyTorch to JAX
    def transfer_weights(pytorch_dict):
        new_params = {'params': {}}
        for i in range(NUM_LAYERS):
            new_params['params'][f'Dense_{i}'] = {
                'kernel': jnp.array(pytorch_dict[f'mlp.{i}.weight'].detach().numpy().T),
                'bias': jnp.array(pytorch_dict[f'mlp.{i}.bias'].detach().numpy())
            }
        return freeze(new_params)

    # Initialize JAX model parameters
    rng = jax.random.PRNGKey(0)
    _, jax_params = jax_net.init_with_output(rng, jnp.zeros((1, INPUT_SIZE)))

    # Transfer weights from PyTorch to JAX
    jax_params = transfer_weights(pytorch_net.state_dict())

    jax_net.params = jax_params

    return jax_net, jax_params

@jax.jit
def cdf_evaluate_model(params, config, points):
    def apply_model(params, inputs):
        return CDFNet_JAX(hidden_layers=(HIDDEN_SIZE,) * (NUM_LAYERS - 1), output_dims=OUTPUT_SIZE).apply(params, inputs)
    
    # Broadcast config to match the batch size of points
    config_broadcast = jnp.broadcast_to(config, (points.shape[0], config.shape[0]))
    
    # Combine config and points
    inputs = jnp.concatenate([config_broadcast, points], axis=-1)

    # Compute CDF values
    cdf_values = apply_model(params, inputs)
    
    # Compute gradients with respect to all inputs
    def model_output(inputs):
        return apply_model(params, inputs).sum()
    
    gradients = jax.vmap(jax.grad(model_output))(inputs)
    
    return cdf_values, gradients

def visualize_arm_cdf(angles, jax_params, save_path='learned_cdf_visualization.png'):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    
    # Generate points for CDF evaluation
    n_points = 100
    x = np.linspace(-20, 20, n_points)
    y = np.linspace(-20, 20, n_points)
    xx, yy = np.meshgrid(x, y)
    points = jnp.array(np.stack((xx.flatten(), yy.flatten()), axis=-1))
    
    # Calculate CDF values
    start_time = time.time()
    config = jnp.array([angles[0], angles[1], angles[2], angles[3], angles[4],
                        jnp.sin(angles[0]), jnp.sin(angles[1]), jnp.sin(angles[2]), jnp.sin(angles[3]), jnp.sin(angles[4]),
                        jnp.cos(angles[0]), jnp.cos(angles[1]), jnp.cos(angles[2]), jnp.cos(angles[3]), jnp.cos(angles[4])])
    

    
    cdf_values, _ = cdf_evaluate_model(jax_params, config, points)
    total_time = time.time() - start_time
    print(f'Total evaluation time: {total_time:.2f} seconds')
    
    cdf_values = np.array(cdf_values).reshape(n_points, n_points)

    # Create heatmap
    heatmap = ax.imshow(cdf_values, cmap='viridis', extent=[-20, 20, -20, 20], origin='lower', aspect='equal', vmin=-2, vmax=2)
    
    # Plot robot arm
    joint_positions = forward_kinematics(angles)
    current_angle = 0
    
    for i, ((shape_name, shape_points), joint_pos) in enumerate(zip(shapes[:NUM_LINKS], joint_positions)):
        if i < len(angles):
            current_angle += angles[i]
        transformed_shape = transform_shape(shape_points, current_angle, joint_pos)
        ax.fill(*zip(*transformed_shape), alpha=0.5)
        ax.plot(joint_pos[0], joint_pos[1], 'ko', markersize=8)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title('CDF Visualization for Robot Arm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('CDF Value')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")

def main():
    trained_model_path="trained_models/cdf_models/cdf_model_5_256_distance.pt"  # or use cdf_model_5_256_eikonal.pt
    # Load the trained model and convert to JAX parameters
    jax_net, jax_params = load_learned_cdf(trained_model_path)

    # Set arm angles
    angles = np.array([-3*np.pi/4, 0, 0, 0, 0])

    # Visualize the arm CDF
    visualize_arm_cdf(angles, jax_params)

if __name__ == "__main__":
    main()