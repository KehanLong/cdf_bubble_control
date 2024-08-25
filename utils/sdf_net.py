import torch.nn as nn

class SDFNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(SDFNet, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.hidden_layers.append(nn.Linear(hidden_size, 1))  # output the scalar SDF
        self.activation = nn.Softplus()

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.activation(layer(x))
        x = self.hidden_layers[-1](x)  # Remove activation for the last layer
        return x
    

'''
Jax version
'''

from flax import linen as nn

class SDFNet(nn.Module):
    hidden_size: int
    num_layers: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        for _ in range(self.num_layers - 2):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.softplus(x)
        x = nn.Dense(1)(x)  # Remove activation for the last layer
        return x
