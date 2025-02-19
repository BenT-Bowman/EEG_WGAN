import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    

class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# class Generator(nn.Module):
#     def __init__(self, seq_length=500, num_channels = 19):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(100, 256),  
#             nn.GELU(),#(True),
#             nn.Linear(256, 512),             
#             nn.GELU(),#(True),
#             nn.Linear(512, 1024),  
#             nn.GELU(),#(True),
#             nn.Linear(1024, 1024*2),  # Hidden layer
#             nn.GELU(),#(True),
#             nn.Linear(1024*2, num_channels*seq_length), 
#             nn.Tanh()

#             # nn.Tanh()  # Tanh activation to output values between -1 and 1
#         )

#     def forward(self, z):
#         return self.model(z) #.view(z.size(0), -1)


# class Generator(nn.Module):
#     def __init__(self, input_size=100, output_size=9500, num_heads=16, hidden_dim=256):
#         super(Generator, self).__init__()
#         self.encoder_linear = nn.Linear(input_size, hidden_dim)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
#         self.decoder_linear = nn.Linear(hidden_dim, output_size)
#         # self.tanh = nn.Tanh()

#     def forward(self, z):
#         z = F.gelu(self.encoder_linear(z))  # Encoder linear layer with GELU activation
#         z = z.unsqueeze(1)  # Add batch dimension for transformer encoder
#         z = self.encoder(z)  # Multi-head attention encoder
#         z = z.squeeze(1)  # Remove batch dimension
#         z = self.decoder_linear(z)  # Decoder linear layer
#         return z

# class Generator(nn.Module):
#     def __init__(self, seq_length=500, num_channels = 19):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(100, 256),  
#             nn.Linear(256, 512),             
#             nn.Linear(512, 1024),  
#             nn.Linear(1024, 1024*2),  # Hidden layer
#             nn.GELU(),#(True),
#             nn.Dropout(0.1),
#             # nn.LayerNorm(1024),
#             nn.Linear(1024*2, num_channels*seq_length),  # Output layer: Match the flattened data shape
#             Reshape(1, num_channels, seq_length),
#             nn.Conv2d(1, 16, kernel_size=(num_channels, 1), padding=(num_channels//2, 0)),
#             nn.GELU(),
#             nn.Conv2d(16, 16, kernel_size=(1, 19), padding=(0, 19//2)),
#             nn.GELU(),
#             nn.Conv2d(16, 1, kernel_size=(1,1)),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.model(z).view(z.size(0), -1)

#Bayazit
# Generator Model

"""
class Generator(nn.Module):
    def __init__(self, seq_length=500, num_channels=19, latent_vector_size=8032):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            Reshape(1, 1, latent_vector_size),
            nn.ConvTranspose2d(1, 16, kernel_size=(10, 1)),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(1, 5), stride=(1, 4)),
            nn.ELU(),
            nn.ConvTranspose2d(16, 16, kernel_size=(10, 1)),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(1, 9), stride=(1, 4)),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self,x):
        x = self.model(x)
        return x.view(x.size(0), -1)

# Critic Model
class Critic(nn.Module):
    def __init__(self, seq_length=500, num_channels=19):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(num_channels, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=(1, 16), stride = (1, 16)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(496, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 19, 500)  # Adjust the input shape if needed
        return self.model(x)"""

class MinMaxScaler(nn.Module):
    def __init__(self, min_val=-1, max_val=1, eps=1e-8):
        """
        Min-max scales the input data to the range [min_val, max_val].
        :param min_val: Minimum value after scaling
        :param max_val: Maximum value after scaling
        :param eps: Small value to prevent division by zero
        """
        super(MinMaxScaler, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def forward(self, x):
        """
        Min-max normalize each sample independently.
        """
        x_min = x.amin(dim=-1, keepdim=True)  # Find min per sample
        x_max = x.amax(dim=-1, keepdim=True)  # Find max per sample
        x = (x - x_min) / (x_max - x_min + self.eps)  # Normalize to [0,1]
        x = x * (self.max_val - self.min_val) + self.min_val  # Scale to [min_val, max_val]
        return x
    


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.gelu2 = nn.GELU()

        # Projection layer if input and output channels mismatch
        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        if self.projection is not None:
            identity = self.projection(x)

        out = self.conv1(x)
        out = self.gelu1(out)
        out = self.conv2(out)
        out = self.gelu2(out)
        
        return out + identity  # Residual connection

class Generator(nn.Module):
    def __init__(self, seq_length=500, num_channels=19, latent_vector_size=8032):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            Reshape(1, 1, latent_vector_size),
            nn.Conv2d(1, 8, kernel_size=(1, 27), stride=(1, 4)),
            nn.GELU(),
            nn.LayerNorm([8, 1, 2002]),
            nn.Conv2d(8, 8, kernel_size=(1, 3), stride=(1, 2)),
            nn.GELU(),
            nn.LayerNorm([8, 1, 1000]),

            ResidualBlock(8, 16, kernel_size=(1, 9), stride=(1, 2)),
            nn.LayerNorm([16, 1, 500]),
            ResidualBlock(16, 16, kernel_size=(1, 5)),
            nn.LayerNorm([16, 1, 500]),
            ResidualBlock(16, 16, kernel_size=(1, 5)),
            nn.LayerNorm([16, 1, 500]),
            ResidualBlock(16, 16, kernel_size=(1, 3)),

            nn.Conv2d(16, 32, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm([32, 1, 500]),

            nn.ConvTranspose2d(32, 32, kernel_size=(num_channels, 1)),
            nn.GELU(),
            nn.LayerNorm([32, 19, 500]),

            ResidualBlock(32, 32, kernel_size=(19, 1)),
            nn.LayerNorm([32, 19, 500]),
            ResidualBlock(32, 32, kernel_size=(19, 1)),
            nn.LayerNorm([32, 19, 500]),

            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)


import torch.nn.utils.spectral_norm as spectral_norm

class Critic(nn.Module):
    def __init__(self, seq_length=500, num_channels=19):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # nn.Conv2d(1, 8, kernel_size=(num_channels, 1)),
            # nn.BatchNorm2d(8),
            # nn.ELU(),

            nn.Conv2d(1, 16, kernel_size=(1, 19), stride = (1, 9)),
            nn.GELU(),
            nn.LayerNorm([16, 19, 54]),
            nn.Conv2d(16, 16, kernel_size=(1, 9), stride = (1, 5)),
            nn.GELU(),
            nn.LayerNorm([16, 19, 10]),

            nn.Conv2d(16, 32, kernel_size=(1, 3), stride = (1, 3)),
            nn.GELU(),
            nn.LayerNorm([32, 19, 3]),

            ResidualBlock(32, 64, kernel_size=(19, 1)),
            ResidualBlock(64, 64, kernel_size=(19, 1)),
            nn.Conv2d(64, 128, kernel_size=(19,1)), # TODO: Consider ResidualBlock for spatial componetj
            nn.LayerNorm([128, 1, 3]),

            nn.GELU(),
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 19, 500)  # Adjust the input shape if needed
        return self.model(x)

# class Critic(nn.Module):
#     def __init__(self, num_classes=1, num_channels=19, num_samples=500, dropout=0.5):
#         super(Critic, self).__init__()
        
#         # 1. Temporal Convolution
#         self.temporal_conv = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32), bias=False),
#             nn.BatchNorm2d(16)
#         )
        
#         # 2. Depthwise Convolution
#         self.depthwise_conv = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=(num_channels, 1), groups=16, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 4)),
#             nn.Dropout(dropout)
#         )
        
#         # 3. Separable Convolution
#         self.separable_conv = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(1, 16), stride=(1, 1), padding=(0, 8), bias=False),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 8)),
#             nn.Dropout(dropout)
#         )
        
#         # Compute output size after convolutions
#         out_dim = self._get_output_dim(num_channels, num_samples)
        
#         # 4. Fully Connected Layer
#         self.fc = nn.Linear(out_dim, num_classes)

#     def _get_output_dim(self, num_channels, num_samples):
#         """Helper function to compute final feature map size dynamically."""
#         x = torch.randn(1, 1, num_channels, num_samples)
#         x = self.temporal_conv(x)
#         x = self.depthwise_conv(x)
#         x = self.separable_conv(x)
#         return x.numel() // x.shape[0]  # Total feature size

#     def forward(self, x):
#         x = x.view(x.size(0), 1, 19, 500)  # Adjust the input shape if needed
#         x = self.temporal_conv(x)
#         x = self.depthwise_conv(x)
#         x = self.separable_conv(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc(x)
#         return x

# class Critic(nn.Module):
#     def __init__(self, seq_length=500, num_channels=19):
#         super(Critic, self).__init__()
#         self.seq_length = seq_length
#         self.num_channels = num_channels
#         self.model = nn.Sequential(
#             nn.Linear(num_channels*seq_length, 1024*2),  # Input layer: Match the flattened data shape
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024*2, 1024),  # Hidden layer
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Linear(1024, 512),  # Hidden layer
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),  # Hidden layer
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1),  # Output layer
#         )

#     def forward(self, x):
#         if x.size(1) != self.num_channels*self.seq_length:
#             x = x.view(x.size(0), -1)
#         return self.model(x)
    

# class Generator(nn.Module):
#     def __init__(self, z_dim: int = 100, out_channels: int = 1, filters: list = [16, 16, 8], kernel_size: int = 6, num_samples: int = 500):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(z_dim, filters[0] * num_samples // 4),
#             nn.GELU(),
#             nn.Unflatten(1, (1, filters[0], num_samples // 4)),
            

#             nn.ConvTranspose2d(1, out_channels=filters[1], kernel_size=(1, kernel_size), stride=(1, 2), padding=(0, kernel_size // 2-1)),
#             nn.GELU(),
#             # nn.BatchNorm2d(filters[1]),

#             nn.ConvTranspose2d(filters[1], out_channels=filters[2], kernel_size=(19, 1), stride=(1, 1), groups=filters[2]),
#             nn.GELU(),
#             # nn.BatchNorm2d(filters[2]),

#             nn.ConvTranspose2d(filters[2], out_channels=filters[0], kernel_size=(1, kernel_size), stride=(1, 2), padding=(0, kernel_size // 2-1)),
#             nn.GELU(),
#             # nn.BatchNorm2d(filters[0]),

#             nn.Conv2d(filters[0], out_channels=1, kernel_size=(16, 1)),
#             # nn.ReLU(),
#             # nn.BatchNorm2d(1),
            
#             nn.Tanh(),
#         )

#     def forward(self, z):
#         # z=z
#         return self.model(z).view(z.size(0), -1)
    



# class Critic(nn.Module):
#     def __init__(self, in_channels:int= 1, filters: list = [8, 16, 16], kernel_size:int = 5,  dropout_rate=0.5, num_classes=1):
#         super().__init__()
#         pool_size, num_samples = 2, 500
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels=filters[0], kernel_size=(1, kernel_size), padding=(0, kernel_size// 2), bias=False),
#             nn.BatchNorm2d(filters[0], False),
#             self._regularization(pool_size, dropout_rate),

#             nn.Conv2d(filters[0], out_channels=filters[1], kernel_size=(in_channels, 1),groups=filters[0], bias=False),
#             nn.BatchNorm2d(filters[1], False),
#             self._regularization(pool_size, dropout_rate),

#             nn.Conv2d(filters[1], out_channels= filters[2], kernel_size=(1, 1), bias=False),
#             nn.BatchNorm2d(filters[2], False),
#             self._regularization(pool_size, dropout_rate),

#             nn.Conv2d(filters[2], out_channels= filters[2], kernel_size=(1, 16), padding=(0, 8), bias=False),
#             nn.BatchNorm2d(filters[2], False),
#             self._regularization(pool_size, dropout_rate),

#             nn.Flatten(),

#             nn.Linear(9424, num_classes)
#         )
#         # print(filters[2] * ((num_samples // pool_size) // pool_size))
#     def _regularization(self, pool_size, dropout_rate):
#         return nn.Sequential(
#             nn.ELU(),
#             nn.AvgPool2d((1, pool_size)),
#             nn.Dropout(dropout_rate),
#             )

#     def forward(self, x):
#         x=x.view(x.size(0), 1, 19, 500)
#         return self.model(x)

if __name__ == "__main__":
    latent_vector_size = 251*32
    noise = torch.randn(32,  latent_vector_size)
    g = Generator(latent_vector_size=latent_vector_size)
    c = Critic()
    print((out:=g(noise)).shape)
    print(c(out).shape)


