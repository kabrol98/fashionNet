import numpy as np 
from torch import nn
# Define AutoEncoder:

class Linspace(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        # Evenly spaced dimensions based on encoder_size.
        linspace_dims = np.arange(
            input_dim, 
            output_dim, 
            step=(output_dim - input_dim) // num_layers)
        assert len(linspace_dims) == num_layers, (len(linspace_dims), num_layers)
        linspace_dims = np.append(linspace_dims, output_dim)
        linspace = []
        for i in range(num_layers):
            mid_layer = nn.Sequential(
                nn.Linear(linspace_dims[i], linspace_dims[i+1]),
                nn.ReLU()
            )
            linspace.append(mid_layer)
        print(linspace)
        self.full_model = nn.Sequential(*linspace)

    def forward(self, x):
        x = self.full_model(x)
        return x


# Define trial linear network
class AutoEncoder(nn.Module):
    
    def __init__(
            self, 
            input_dim, 
            encoder_size,
            decoder_size,
            n_layers, 
            layer_dim):
        super().__init__()
        self.Encoder = Linspace(input_dim, layer_dim, encoder_size)
        self.Decoder = Linspace(layer_dim, input_dim, decoder_size)
        
        # self.body = [nn.Sequential(
        #     nn.Linear(layer_dim, layer_dim), 
        #     nn.ReLU()) for _ in range(n_layers)]
        
        # self.full_model = nn.Sequential(self.Encoder, *self.body, self.Decoder)
        self.full_model = nn.Sequential(self.Encoder, self.Decoder)

    def forward(self, x):
        x = self.full_model(x)
        return x

# Define Trial CNN
class KCNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, layer_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.intro = nn.Sequential(
            nn.Linear(500, layer_dim),
            nn.ReLU()
        )
        self.body = [nn.Sequential(
            nn.Linear(layer_dim, layer_dim), 
            nn.ReLU()) for _ in range(n_layers)]
        self.out = nn.Sequential(
            nn.Linear(layer_dim, output_dim),
            nn.Softmax(dim=1)
        )
        self.all_layers = [self.conv1, self.conv2, self.flatten, self.intro, *self.body, self.out]

    def forward(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x
        