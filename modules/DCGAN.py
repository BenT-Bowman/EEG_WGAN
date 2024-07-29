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

class Generator(nn.Module):
    def __init__(self, seq_length=500, num_channels = 19):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.Linear(256, num_channels*seq_length),  # Output layer: Match the flattened data shape
            Reshape(1, num_channels, seq_length),
            nn.Conv2d(1, 32, kernel_size=(num_channels, 1), padding=(num_channels//2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 19), padding=(0, 19//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(num_channels, 1), padding=(num_channels//2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(1, 19), padding=(0, 19//2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=(1,1)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), -1)

class Critic(nn.Module):
    def __init__(self, seq_length=500, num_channels = 19):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(num_channels, 1), padding=(num_channels//2, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 19), padding=(0, 19//2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=(num_channels, 1), padding=(num_channels//2, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=(1, 19), padding=(0, 19//2)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32*seq_length*num_channels, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x=x.view(x.size(0), 1, 19, 500)
        return self.model(x)

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
    noise = torch.randn(100,  100)
    g = Generator()
    c = Critic()
    print((out:=g(noise)).shape)
    print(c(out).shape)


