
import jax.numpy as jnp
from flax import linen as nn

class RobotArmNet(nn.Module):
    hidden_size: int
    output_size: int
    num_layers: int = 5

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.softplus(x)
        for _ in range(self.num_layers - 2):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.softplus(x)
        x = nn.Dense(self.output_size)(x)
        return x