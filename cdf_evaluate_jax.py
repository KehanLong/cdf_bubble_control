import numpy as np
import matplotlib.pyplot as plt
import torch
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
import re

from utils.cdf_net import CDF_Net, CDFNet_JAX




def initialize_model(model, input_shape):
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones(input_shape))
    return params

def transfer_weights(torch_model, jax_model, jax_params):
    torch_state_dict = torch_model.state_dict()
    jax_params = unfreeze(jax_params)

    for i in range(len(jax_model.hidden_dims) + 1):
        torch_weight = torch_state_dict[f'lin{i}.weight'].numpy()
        torch_bias = torch_state_dict[f'lin{i}.bias'].numpy()

        jax_params['params'][f'Dense_{i}']['kernel'] = torch_weight.T
        jax_params['params'][f'Dense_{i}']['bias'] = torch_bias

    return freeze(jax_params)

def load_learned_cdf_to_jax(trained_model_path="trained_models/cdf_models/cdf_model_2_links.pt"):
    num_links = infer_num_links(trained_model_path)
    
    torch_model = CDF_Net(
        input_dims=num_links * 3 + 2,
        hidden_dims=[512, 512, 512, 512],
        output_dims=1,
        skip_in=(4,),
    )
    torch_model.load_state_dict(torch.load(trained_model_path, map_location='cpu'))
    torch_model.eval()

    jax_model = CDFNet_JAX(
        input_dims=num_links * 3 + 2,
        hidden_dims=[512, 512, 512, 512],
        output_dims=1,
        skip_in=(4,)
    )

    # Initialize JAX model with dummy input
    dummy_input = jnp.ones((1, num_links * 3 + 2))
    initial_params = initialize_model(jax_model, dummy_input.shape)

    # Transfer weights
    jax_params = transfer_weights(torch_model, jax_model, initial_params)

    return jax_model, jax_params

@jax.jit
def cdf_evaluate_model_jax(params, config, point):
    def apply_model(params, inputs, num_links):
        return CDFNet_JAX(
            input_dims=num_links * 3 + 2,
            hidden_dims=[512, 512, 512, 512],
            output_dims=1,
            skip_in=(4,)
        ).apply(params, inputs)
    
    # Encode configurations
    encoded_config = jnp.concatenate([config, jnp.sin(config), jnp.cos(config)], axis=-1)

    num_links = config.shape[-1]  # Assuming config is (N, num_links) or (num_links,)
    
    # Ensure encoded_config is 2D and broadcast to match point's first dimension
    if encoded_config.ndim == 1:
        encoded_config = encoded_config[jnp.newaxis, :]
    encoded_config = jnp.broadcast_to(encoded_config, (point.shape[0], encoded_config.shape[-1]))
    
    # Combine encoded config and point
    inputs = jnp.concatenate([encoded_config, point], axis=-1)

    # Compute CDF value
    cdf_value = apply_model(params, inputs, num_links).squeeze()
    
    # Compute gradient with respect to inputs
    grad_fn = jax.grad(lambda x: apply_model(params, x, num_links).sum())
    gradient = grad_fn(inputs)
    
    return cdf_value, gradient

def infer_num_links(trained_model_path):
    match = re.search(r'(\d+)_links', trained_model_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Number of links not found in the model path.")

def plot_cdf_field(ax, model, params, joint_pair, num_links, obstacle, resolution=50):
    theta = jnp.linspace(-jnp.pi, jnp.pi, resolution)
    Theta1, Theta2 = jnp.meshgrid(theta, theta)
    
    # Create a grid of configurations
    configs = jnp.zeros((resolution * resolution, num_links))
    configs = configs.at[:, joint_pair[0]].set(Theta1.ravel())
    configs = configs.at[:, joint_pair[1]].set(Theta2.ravel())
    # Set fixed angles for other joints
    fixed_joints = [j for j in range(num_links) if j not in joint_pair]
    configs = configs.at[:, fixed_joints].set(jnp.pi / 4)

    # Evaluate CDF for all configurations at once
    points = jnp.tile(obstacle[:2], (resolution * resolution, 1))
    cdf_values, _ = cdf_evaluate_model_jax(params, configs, points)

    Z = cdf_values.reshape(resolution, resolution)

    # Plot CDF
    contour = ax.contourf(Theta1, Theta2, Z, levels=20, cmap='viridis')
    zero_level = ax.contour(Theta1, Theta2, Z, levels=[0.1], colors='r', linewidths=2)
    plt.colorbar(contour, ax=ax, label='CDF Value')

    # Add labels to the zero level set
    ax.clabel(zero_level, inline=True, fmt='Zero', colors='r')

    ax.set_xlabel(f'θ{joint_pair[0] + 1}')
    ax.set_ylabel(f'θ{joint_pair[1] + 1}')
    ax.set_title(f'CDF Field for Joint Pair {joint_pair}')

def main():
    trained_model_path = "trained_models/cdf_models/cdf_model_2_links.pt"
    jax_model, jax_params = load_learned_cdf_to_jax(trained_model_path)

    obstacle = jnp.array([1.5, 1.5, 0])  # Example obstacle position

    num_links = infer_num_links(trained_model_path)
    joint_pairs = [(i, j) for i in range(num_links) for j in range(i + 1, num_links)]
    num_plots = len(joint_pairs)

    cols = 2
    rows = (num_plots + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axs = axs.ravel()

    for i, joint_pair in enumerate(joint_pairs):
        plot_cdf_field(axs[i], jax_model, jax_params, joint_pair, num_links, obstacle)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig('cdf_field_jax.png')
    plt.close()

    print("CDF field plot saved as 'cdf_field_jax.png'")

if __name__ == "__main__":
    main()
