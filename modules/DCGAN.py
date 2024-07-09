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

class Generator(nn.Module):
    def __init__(self, seq_length=500, num_channels = 19):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),  
            nn.Linear(256, 512),             
            nn.Linear(512, 1024),  
            nn.Linear(1024, 1024*2),  # Hidden layer
            nn.GELU(),#(True),
            nn.Dropout(0.1),
            # nn.LayerNorm(1024),
            nn.Linear(1024*2, num_channels*seq_length),  # Output layer: Match the flattened data shape
            Reshape(1, num_channels, seq_length),
            nn.Conv2d(1, 16, kernel_size=(num_channels, 1), padding=(num_channels//2, 0)),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=(1, 19), padding=(0, 19//2)),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=(1,1)),
            nn.Tanh()

            # nn.Tanh()  # Tanh activation to output values between -1 and 1
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), -1)





# Discriminator Model
class Critic(nn.Module):
    def __init__(self, seq_length=500, num_channels=19):
        super(Critic, self).__init__()
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.model = nn.Sequential(
            nn.Linear(num_channels*seq_length, 1024*2),  # Input layer: Match the flattened data shape
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024*2, 1024),  # Hidden layer
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),  # Hidden layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Hidden layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),  # Output layer
        )

    def forward(self, x):
        if x.size(1) != self.num_channels*self.seq_length:
            x = x.view(x.size(0), -1)
        return self.model(x)

if __name__ == "__main__":
    noise = torch.randn(100,  100)
    g = Generator()
    c = Critic()
    print((out:=g(noise)).shape)
    print(c(out).shape)

if __name__ == "__main__":
    noise = torch.randn(100,  100)
    g = Generator()
    c = Critic()
    print((out:=g(noise)).shape)
    print(c(out).shape)