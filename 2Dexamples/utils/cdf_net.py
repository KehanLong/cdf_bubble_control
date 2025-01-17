from torch import nn
import numpy as np
import torch

class CDF_Net(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims = [512, 512, 512, 512],
        skip_in=(2, 4),
        use_skip_connections=True
    ):
        super(CDF_Net, self).__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        dims = [input_dims] + hidden_dims + [output_dims]
        
        self.num_layers = len(dims)
        self.skip_in = skip_in if use_skip_connections else ()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - input_dims
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
            setattr(self, f"lin{layer}", lin)
            
            # if layer < self.num_layers - 2:
            #     bn = nn.LayerNorm(out_dim)
            #     setattr(self, f"bn{layer}", bn)

        # Different activation options - uncomment one at a time to test
        self.activation = nn.GELU()
        #self.activation = nn.Tanh()
        #self.activation = nn.Mish()
        #self.activation = nn.SiLU()
        #self.activation = nn.ReLU()
        
    def forward(self, x):
        input = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                x = torch.cat([x, input], -1)
            x = lin(x)
            
            if layer < self.num_layers - 2:
                #bn = getattr(self, f"bn{layer}")
                #x = bn(x)
                x = self.activation(x)
        
        # Apply absolute value to ensure positive distances
        x = torch.abs(x)
        return x



import jax.numpy as jnp
from flax import linen as lnn
from typing import Sequence, Tuple

class CDFNet_JAX(lnn.Module):
    input_dims: int
    hidden_dims: Sequence[int]
    output_dims: int = 1
    skip_in: Tuple[int] = (2,4)
    use_skip_connections: bool = True

    @lnn.compact
    def __call__(self, x):
        input_x = x
        dims = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        num_layers = len(dims)

        for layer in range(num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - self.input_dims
            else:
                out_dim = dims[layer + 1]
            
            x = lnn.Dense(out_dim)(x)

            if layer < num_layers - 2:
                x = lnn.relu(x)
            
            if layer + 1 in self.skip_in:
                x = jnp.concatenate([x, input_x], axis=-1)

        return x




