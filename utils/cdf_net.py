from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

class CDF_Net(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims=1,
        skip_in=(4,),
        geometric_init=True,
        radius_init=1,
        beta=100,
        use_skip_connections=True
    ):
        super(CDF_Net, self).__init__()
        
        self.input_dims = input_dims
        dims = [input_dims] + hidden_dims + [output_dims]
        
        self.num_layers = len(dims)
        self.skip_in = skip_in if use_skip_connections else ()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - input_dims  # Adjust output dimension for skip connections
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(lin.bias, 0)

            setattr(self, f"lin{layer}", lin)

        self.activation = nn.Softplus(beta=beta) if beta > 0 else nn.ReLU()
        
    def forward(self, x):
        input = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                x = torch.cat([x, input], -1)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x  



import jax
import jax.numpy as jnp
from flax import linen as lnn
from typing import Sequence, Tuple, Any

class CustomDense(lnn.Module):
    features: int
    use_bias: bool = True

    @lnn.compact
    def __call__(self, inputs):
        kernel = self.variable('params', 'kernel', lambda: None)
        bias = self.variable('params', 'bias', lambda: None)
        print(f"CustomDense input shape: {inputs.shape}, kernel shape: {kernel.value.shape}")
        y = jnp.dot(inputs, kernel.value.T)
        if self.use_bias:
            y += bias.value
        print(f"CustomDense output shape: {y.shape}")
        return y

class CDFNet_JAX(lnn.Module):
    input_dims: int
    hidden_dims: Sequence[int]
    output_dims: int = 1
    skip_in: Tuple[int] = (4,)
    beta: float = 100
    use_skip_connections: bool = True

    @staticmethod
    def custom_softplus(x, beta):
        return (1.0 / beta) * jnp.log(1 + jnp.exp(beta * x))

    @lnn.compact
    def __call__(self, x):
        print(f"Input shape: {x.shape}")
        input_x = x
        dims = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        num_layers = len(dims)
        skip_in = self.skip_in if self.use_skip_connections else ()

        for layer in range(num_layers - 1):
            print(f"Layer {layer}, input shape: {x.shape}")
            
            out_dim = dims[layer + 1]
            lin = CustomDense(features=out_dim)
            x = lin(x)

            if layer < num_layers - 2:
                x = self.custom_softplus(x, self.beta) if self.beta > 0 else jax.nn.relu(x)
            
            if layer + 1 in skip_in:
                x = jnp.concatenate([x, input_x], axis=-1)
                print(f"After skip connection, shape: {x.shape}")
            
            print(f"Layer {layer} output shape: {x.shape}")

        return x




