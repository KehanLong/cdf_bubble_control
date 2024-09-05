from torch import nn
import torch.nn.functional as F

class CDFNet(nn.Module):
    def __init__(self, input_dims=17, output_dims=1, hidden_layers=[128, 128, 128, 128, 128]):
        super(CDFNet, self).__init__()
        
        layers = [input_dims] + hidden_layers + [output_dims]
        
        self.mlp = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.mlp.append(nn.Linear(layers[i], layers[i+1]))
        
        self.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:  # Apply softplus to all but the last layer
                x = F.softplus(x)
        
        return F.relu(x)  # Apply ReLU to the final output

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)



from flax import linen as jnn
import jax.numpy as jnp
from typing import Sequence


class CDFNet_JAX(jnn.Module):
    hidden_layers: Sequence[int] = (128, 128, 128, 128, 128)
    output_dims: int = 1

    @jnn.compact
    def __call__(self, x):
        for units in self.hidden_layers:
            x = jnn.Dense(units)(x)
            x = jnn.softplus(x)
        
        x = jnn.Dense(self.output_dims)(x)
        return jnp.maximum(x, 0)  # ReLU activation

    @staticmethod
    def init_weights(key, shape, dtype):
        return jnp.array(jnn.initializers.xavier_uniform()(key, shape, dtype))

    @staticmethod
    def init_bias(key, shape, dtype):
        return jnp.zeros(shape, dtype)