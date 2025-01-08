import torch
import torch.nn as nn

class CDFNetwork(nn.Module):
    def __init__(
        self,
        input_dims=21,  # 6 (joints) * 3 (original + sin + cos) + 3 (3D point)
        output_dims=1,
        hidden_dims=[1024, 512, 256, 128, 128],
        skip_in=(2, 4),
        use_skip_connections=True
    ):
        super(CDFNetwork, self).__init__()
        
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
                x = self.activation(x)
        
        # Apply absolute value to ensure positive distances
        x = torch.abs(x)
        return x 