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
    
# Define a small conv fusion block to process concatenated features.
# class FusionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(FusionBlock, self).__init__()
#         # padding so that spatial dims are preserved
#         self.conv = spectral_norm(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
#         )
#         self.gelu = nn.GELU()
#     def forward(self, x):
#         return self.gelu(self.conv(x))

# class Generator(nn.Module):
#     def __init__(self, seq_length=1025, num_channels=19, latent_vector_size=4100):
#         """
#         We assume the input to the UNet will have shape [B, 1, 19, 1025].
#         We downsample only along the width dimension (1025) while keeping 19 intact.
#         """
#         super(Generator, self).__init__()
        
#         # ================================================================
#         # Encoder
#         # ================================================================
#         # Block 1: from [B,1,19,1025] -> [B,8,19,257]
#         # Using kernel=(1,27), stride=(1,4) and padding to preserve height.
#         self.enc1_conv = spectral_norm(
#             nn.Conv2d(1, 8, kernel_size=(1,27), stride=(1,4), padding=(0,13))
#         )
#         self.enc1_activation = nn.GELU()
        
#         # Block 2: [B,8,19,257] -> [B,16,19,129]
#         self.enc2_conv = spectral_norm(
#             nn.Conv2d(8, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1))
#         )
#         self.enc2_activation = nn.GELU()
        
#         # Block 3: [B,16,19,129] -> [B,32,19,65]
#         self.enc3_res = ResidualBlock(16, 32, kernel_size=(1,9), stride=(1,2))
        
#         # Block 4: [B,32,19,65] -> [B,64,19,33]
#         self.enc4_res = ResidualBlock(32, 64, kernel_size=(1,3), stride=(1,2))
        
#         # ================================================================
#         # Bottleneck
#         # ================================================================
#         # [B,64,19,33] -> [B,128,19,17]
#         self.bottleneck_conv = spectral_norm(
#             nn.Conv2d(64, 128, kernel_size=(1,3), stride=(1,2), padding=(0,1))
#         )
#         self.bottleneck_activation = nn.GELU()
        
#         # ================================================================
#         # Decoder
#         # ================================================================
#         # Decoder block 1:
#         # Upconv: [B,128,19,17] -> [B,64,19,33]
#         self.dec1_upconv = spectral_norm(
#             nn.ConvTranspose2d(128, 64, kernel_size=(1,3), stride=(1,2), padding=(0,1))
#         )
#         self.dec1_activation = nn.GELU()
#         # Fusion with encoder 4 ([B,64,19,33] skip connection)
#         # After concatenation, channels become 64+64 = 128, then fuse down to 64.
#         self.dec1_fusion = FusionBlock(128, 64)
        
#         # Decoder block 2:
#         # Upconv: [B,64,19,33] -> [B,32,19,65]
#         self.dec2_upconv = spectral_norm(
#             nn.ConvTranspose2d(64, 32, kernel_size=(1,3), stride=(1,2), padding=(0,1))
#         )
#         self.dec2_activation = nn.GELU()
#         # Skip connection from encoder 3 ([B,32,19,65]), so fuse 32+32 -> 32.
#         self.dec2_fusion = FusionBlock(64, 32)
        
#         # Decoder block 3:
#         # Upconv: [B,32,19,65] -> [B,16,19,129]
#         self.dec3_upconv = spectral_norm(
#             nn.ConvTranspose2d(32, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1))
#         )
#         self.dec3_activation = nn.GELU()
#         # Skip connection from encoder 2 ([B,16,19,129]), fuse 16+16 -> 16.
#         self.dec3_fusion = FusionBlock(32, 16)
        
#         # Decoder block 4:
#         # Upconv: [B,16,19,129] -> [B,8,19,257]
#         self.dec4_upconv = spectral_norm(
#             nn.ConvTranspose2d(16, 8, kernel_size=(1,3), stride=(1,2), padding=(0,1))
#         )
#         self.dec4_activation = nn.GELU()
#         # Skip connection from encoder 1 ([B,8,19,257]), fuse 8+8 -> 8.
#         self.dec4_fusion = FusionBlock(16, 8)
        
#         # Decoder block 5 (Final upsampling):
#         # Upconv: [B,8,19,257] -> [B,1,19,1025]
#         self.dec5_upconv = spectral_norm(
#             nn.ConvTranspose2d(8, 1, kernel_size=(1,27), stride=(1,4), padding=(0,13))
#         )
#         # Use Tanh to get output in range [-1, 1]
#         self.out_activation = nn.Tanh()

#     def forward(self, x):
#         # x shape: [B,1,19,1025]
#         # ---------------------------
#         # Encoder
#         enc1 = self.enc1_activation(self.enc1_conv(x))  
#         # enc1 shape: [B,8,19,257]
        
#         enc2 = self.enc2_activation(self.enc2_conv(enc1))
#         # enc2 shape: [B,16,19,129]
        
#         enc3 = self.enc3_res(enc2)
#         # enc3 shape: [B,32,19,65]  (ResidualBlock includes its own activation)
        
#         enc4 = self.enc4_res(enc3)
#         # enc4 shape: [B,64,19,33]
        
#         # ---------------------------
#         # Bottleneck
#         bottleneck = self.bottleneck_activation(self.bottleneck_conv(enc4))
#         # bottleneck shape: [B,128,19,17]
        
#         # ---------------------------
#         # Decoder
#         dec1 = self.dec1_activation(self.dec1_upconv(bottleneck))
#         # dec1 shape: [B,64,19,33]
#         # Skip connection from encoder 4:
#         dec1 = torch.cat([dec1, enc4], dim=1)  # shape: [B,64+64=128,19,33]
#         dec1 = self.dec1_fusion(dec1)           # fuse to 64 channels
        
#         dec2 = self.dec2_activation(self.dec2_upconv(dec1))
#         # dec2 shape: [B,32,19,65]
#         dec2 = torch.cat([dec2, enc3], dim=1)    # [B,32+32=64,19,65]
#         dec2 = self.dec2_fusion(dec2)            # fuse to 32 channels
        
#         dec3 = self.dec3_activation(self.dec3_upconv(dec2))
#         # dec3 shape: [B,16,19,129]
#         dec3 = torch.cat([dec3, enc2], dim=1)    # [B,16+16=32,19,129]
#         dec3 = self.dec3_fusion(dec3)            # fuse to 16 channels
        
#         dec4 = self.dec4_activation(self.dec4_upconv(dec3))
#         # dec4 shape: [B,8,19,257]
#         dec4 = torch.cat([dec4, enc1], dim=1)    # [B,8+8=16,19,257]
#         dec4 = self.dec4_fusion(dec4)            # fuse to 8 channels
        
#         # Final upsampling to get back to full width 1025
#         out = self.dec5_upconv(dec4)
#         out = self.out_activation(out)
#         # out shape: [B,1,19,1025]
#         return out
class Generator(nn.Module):
    def __init__(self, seq_length=1025, num_channels=19, latent_vector_size=4100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            Reshape(1, 1, latent_vector_size),
            
            nn.Conv2d(1, 8, kernel_size=(1, 27), stride=(1, 4)),
            nn.GELU(),
            nn.LayerNorm([8, 1, 4102]),
            
            nn.Conv2d(8, 16, kernel_size=(1, 3), stride=(1, 2)),
            nn.GELU(),
            nn.LayerNorm([16, 1, 2050]),
            
            ResidualBlock(16, 32, kernel_size=(1, 9), stride=(1, 2)),
            nn.LayerNorm([32, 1, 1025]),
            
            nn.ConvTranspose2d(32, 32, kernel_size=(num_channels, 1)),
            nn.GELU(),
            ResidualBlock(32, 32, kernel_size=(19, 1)),
            nn.LayerNorm([32, 19, 1025]),
            
            ResidualBlock(32, 32, kernel_size=(1, 21)),
            nn.LayerNorm([32, 19, 1025]),
            
            ResidualBlock(32, 64, kernel_size=(9, 1)),
            nn.LayerNorm([64, 19, 1025]),
            
            ResidualBlock(64, 64, kernel_size=(1, 19)),
            nn.LayerNorm([64, 19, 1025]),
            
            ResidualBlock(64, 64, kernel_size=(19, 1)),
            nn.LayerNorm([64, 19, 1025]),
            
            ResidualBlock(64, 64, kernel_size=(1, 9)),
            nn.LayerNorm([64, 19, 1025]),
            
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


 

import torch.nn.utils.spectral_norm as spectral_norm

class Critic(nn.Module):
    def __init__(self, seq_length=1025, num_channels=19):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(1, 8, kernel_size=(1, 27)),
            nn.LayerNorm([8, num_channels, seq_length]),
            nn.AvgPool2d(kernel_size=(1, 2)),
            
            ResidualBlock(8, 8, kernel_size=(1, 19)),
            nn.LayerNorm([8, num_channels, seq_length // 2]),
            nn.AvgPool2d(kernel_size=(1, 2)),
            
            ResidualBlock(8, 16, kernel_size=(1, 9)),
            nn.LayerNorm([16, num_channels, seq_length // 4]),
            nn.AvgPool2d(kernel_size=(1, 2)),
            
            ResidualBlock(16, 32, kernel_size=(19, 1)),
            nn.LayerNorm([32, num_channels, seq_length // 8]),
            
            ResidualBlock(32, 32, kernel_size=(9, 1)),
            nn.LayerNorm([32, num_channels, seq_length // 8]),
            
            ResidualBlock(32, 32, kernel_size=(3, 1)),
            nn.LayerNorm([32, num_channels, seq_length // 8]),
            
            nn.Conv2d(32, 32, kernel_size=(19, 1)),
            nn.GELU(),
            nn.LayerNorm([32, 1, seq_length // 8]),
            
            ResidualBlock(32, 32, kernel_size=(1, 5)),
            nn.LayerNorm([32, 1, seq_length // 8]),
            
            nn.Flatten(),
            nn.Linear(32 * (seq_length // 8), 128),
            nn.GELU(),
            nn.LayerNorm([128]),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.LayerNorm([16]),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 19, -1)  # Ensure input shape is correct
        return self.model(x)

# class Critic(nn.Module):
#     def __init__(self, seq_length=1025, num_channels=19):
#         super(Critic, self).__init__()
#         self.model = nn.Sequential(
#             ResidualBlock(1, 8, kernel_size=(1, 27)),
#             nn.LayerNorm([8, 19, seq_length]),
#             nn.AvgPool2d(kernel_size=(1, 2)),

#             ResidualBlock(8, 8, kernel_size=(19, 1)),
#             nn.LayerNorm([8, 19, seq_length//2]),

#             ResidualBlock(8, 8, kernel_size=(1, 19)),
#             nn.LayerNorm([8, 19, seq_length//2]),
#             nn.AvgPool2d(kernel_size=(1, 2)),

#             ResidualBlock(8, 16, kernel_size=(19, 1)),
#             nn.LayerNorm([16, 19, seq_length//4]),

#             ResidualBlock(16, 16, kernel_size=(1, 9)),
#             nn.LayerNorm([16, 19, seq_length//4]),
#             nn.AvgPool2d(kernel_size=(1, 2)),

#             ResidualBlock(16, 16, kernel_size=(19, 1)),
#             nn.LayerNorm([16, 19, (seq_length//8)]),

#             nn.Conv2d(16, 16, kernel_size=(19,1)), # TODO: Consider ResidualBlock for spatial componetj
#             nn.GELU(),
#             nn.LayerNorm([16, 1, (seq_length//8)]),
#             nn.AvgPool2d(kernel_size=(1, 2)),

#             ResidualBlock(16, 16, kernel_size=(1, 5)),
#             nn.LayerNorm([16, 1, (seq_length//16)]),

#             nn.Flatten(),
#             nn.Linear(16*(seq_length//16), 1),
#             # nn.GELU(),
#             # # nn.LayerNorm([128]),

#             # nn.Linear(128, 16),
#             # nn.GELU(),
#             # # nn.LayerNorm([16]),

#             # nn.Linear(16, 1),
#         )

#     def forward(self, x):
#         x = x.view(x.size(0), 1, 19, -1)  # Adjust the input shape if needed
#         return self.model(x)

if __name__ == "__main__":
    print((8016+200)*2)
    latent_vector_size = (8016+200)*2
    noise = torch.randn(32,  1, latent_vector_size)
    g = Generator(latent_vector_size=latent_vector_size)
    c = Critic()
    print((out:=g(noise)).shape)
    print(c(out).shape)


