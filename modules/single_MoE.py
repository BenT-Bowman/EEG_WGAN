import torch
import torch.nn as nn
import torch.nn.functional as F

class single_channel_gen(nn.Module):
    def __init__(self, latent_vector_size = 2000):
        super().__init__()
        self.latent_vector_size = latent_vector_size
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(1, 15), padding=(0, 15//2))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(1, 9), padding= (0, 9//2))
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(1, 5), padding= (0, 5//2))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(1, 3), padding= (0, 3//2), stride=(1, 2))
        self.final = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        x = x.view(x.shape[0], 1, 1, latent_vector_size)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.tanh(self.final(x))
        return x

class single_channel_critic(nn.Module):
    def __init__(self, ):
        ...
    def forward(self, x):
        ...

class gen_combo_layer(nn.Module):
    def __init__(self, ):
        ...
    def forward(self, x):
        ...

class critic_combo_layer(nn.Module):
    def __init__(self, ):
        ...
    def forward(self, x):
        ...


class Generator(nn.Module):
    def __init__(self, ):
        ...
    def forward(self, x):
        ...

class Critic(nn.Module):
    def __init__(self, ):
        ...
    def forward(self, x):
        ...


if __name__ == "__main__":
    latent_vector_size = 2000
    noise = torch.randn(32,  latent_vector_size)
    g = single_channel_gen(latent_vector_size=latent_vector_size)
    print((out:=g(noise)).shape)


