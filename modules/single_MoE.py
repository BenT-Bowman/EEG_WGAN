import torch
import torch.nn as nn
import torch.nn.functional as F

class single_channel_gen(nn.Module):
    def __init__(self, latent_vector_size = 1000):
        super().__init__()
        self.latent_vector_size = latent_vector_size
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(1, 15), padding=(0, 15//2))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(1, 9), padding= (0, 9//2))
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(1, 5), padding= (0, 5//2))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(1, 3), padding= (0, 3//2), stride=(1, 2))
        self.final = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        x = x.view(x.shape[0], 1, 1, self.latent_vector_size)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.tanh(self.final(x))
        return x

class single_channel_critic(nn.Module):
    def __init__(self, input_size=500, is_single_channel = True):
        super().__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 15), padding=(0, 15//2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 9), padding= (0, 9//2), stride=(1, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 5), padding= (0, 5//2), stride=(1, 2))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding= (0, 3//2), stride=(1, 2))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding= (0, 3//2), stride=(1, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.is_single_channel = is_single_channel
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.shape[0], 1, 1, self.input_size)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        if self.is_single_channel:
            x = self.fc(x)
        return x
    

class ElectrodeWiseConv(nn.Module):
    def __init__(self, num_electrodes:int, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0):
        super().__init__()
        self.num_electrodes = num_electrodes

        assert isinstance(kernel_size, int), "Kernel Size only accepts integers"
        assert isinstance(stride, int), "Stride only accepts integers"
        assert isinstance(padding, int), "Padding only accepts integers"

        self.sub_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding))
            ) for _ in range(num_electrodes)
        ])
        
    def forward(self, x):
        assert x.shape[2] == self.num_electrodes
        futures = [torch.jit.fork(self.sub_modules[i], x[:, :, i, :].unsqueeze(2)) for i in range(self.num_electrodes)]
        outputs = [torch.jit.wait(f) for f in futures]

        x = torch.cat(outputs, dim=2)  # [B, 1, E, T']
        return x
    

class Generator(nn.Module):
    def __init__(self, num_electrodes=19, latent_vector_size=2000):
        super().__init__()
        self.latent_dim = latent_vector_size
        self.ewconv1 = ElectrodeWiseConv(1, 1, 8, 3, padding=1, stride=2)

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(9, 1)),
            nn.LeakyReLU(0.2)
        )

        self.ewconv2 = ElectrodeWiseConv(9, 16, 32, 3, padding=1)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(11, 1)),
            nn.LeakyReLU(0.2)
        )

        self.ewconv3 = ElectrodeWiseConv(19, 32, 64, 3, padding=1, stride=2)
        self.ewconv4 = ElectrodeWiseConv(19, 64, 128, 3, padding=1)

        self.output_layer = nn.Conv2d(128, 1, kernel_size=1)  # Separable 1x1 convolution for final output

    def forward(self, x):
        assert x.shape[1] == self.latent_dim
        x = x.view(x.shape[0], 1, 1, self.latent_dim)

        x = F.leaky_relu(self.ewconv1(x), 0.2)
        x = self.upsample1(x)
        x = F.leaky_relu(self.ewconv2(x), 0.2)
        x = self.upsample2(x)
        x = F.leaky_relu(self.ewconv3(x), 0.2)
        x = self.ewconv4(x)
        
        x = F.tanh(self.output_layer(x))  # Final activation for output scaling
        return x
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

# class Generator(nn.Module):
#     def __init__(self, num_electrodes=19, latent_vector_size=1000, pretrained_gen=None):
#         super().__init__()
#         self.num_electrodes = num_electrodes
#         self.latent_vector_size = latent_vector_size

#         # Learnable embeddings for each electrode
#         self.electrode_embeddings = nn.Parameter(torch.randn(num_electrodes, latent_vector_size) * 0.1)

#         # Create independent generators for each electrode
#         self.generators = nn.ModuleList([
#             single_channel_gen(latent_vector_size) for _ in range(num_electrodes)
#         ])

#         # Load pretrained generator weights and perturb slightly
#         if pretrained_gen is not None:
#             pretrained_state_dict = pretrained_gen.state_dict()
#             for gen in self.generators:
#                 gen.load_state_dict(pretrained_state_dict)
#                 # for param in gen.parameters():
#                 #     param.data += 0.1 * torch.randn_like(param)  # Small perturbation

#             reference_state_dict = self.generators[0].state_dict()
#             all_same = all(
#                 all(torch.equal(param, reference_state_dict[name]) for name, param in gen.state_dict().items())
#                 for gen in self.generators
#             )
#             print("All generators are identical:", all_same)
        
#         self.pre_process = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=( 19, 1), padding=(19//2, 0)),
#             nn.GELU(),
#             nn.Conv2d(16, 16, kernel_size=( 1, 19), padding=(0, 19//2)),
#             nn.GELU(),
#             nn.Conv2d(16, 1, kernel_size=1),
#             nn.GELU(),
#         )

#     def forward(self, x):
#         # Expand latent input: [B, latent_vector_size] → [B, 1, 1, latent_vector_size]
#         x = x.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_electrodes, 1)

#         batch_size, _, num_electrodes, seq_length = x.shape  
#         assert num_electrodes == self.num_electrodes, "Mismatch in electrode count!"

#         x = self.pre_process(x)

#         # Add electrode-specific embeddings to the latent vector
#         x = x + self.electrode_embeddings.unsqueeze(0).unsqueeze(1)  # [B, 1, E, latent_vector_size]

#         # Parallel execution using torch.jit.fork
#         futures = [torch.jit.fork(self.generators[i], x[:, :, i, :].unsqueeze(2)) for i in range(self.num_electrodes)]
#         outputs = [torch.jit.wait(f) for f in futures]

#         x = torch.cat(outputs, dim=2)  # [B, 1, E, T']
#         return x

# class Critic(nn.Module):
#     def __init__(self, num_electrodes=19, input_size=500, aggregation="mean", pretrained_critic=None):
#         super().__init__()
#         self.num_electrodes = num_electrodes
#         self.aggregation = aggregation  
#         self.input_size = input_size

#         # Learnable embeddings per electrode
#         self.electrode_embeddings = nn.Parameter(torch.randn(num_electrodes, 128))  

#         self.critics = nn.ModuleList([
#             single_channel_critic(input_size) for _ in range(num_electrodes)
#         ])

#         if pretrained_critic is not None:
#             pretrained_state_dict = pretrained_critic.state_dict()
#             for i, critic in enumerate(self.critics):
#                 critic.load_state_dict(pretrained_state_dict)
#                 for param in critic.parameters():
#                     param.data += 0.01 * torch.randn_like(param)  # Slight perturbation

#         for critic in self.critics:
#             critic.is_single_channel = False

#         self.fc = nn.Sequential(
#             nn.Linear(128 * num_electrodes, 1),
#         )

#     def forward(self, x):
#         x = x.view(x.shape[0], 1, self.num_electrodes, self.input_size)
#         assert x.shape[2] == self.num_electrodes, "Mismatch in electrode count!"

#         # Compute critic scores
#         futures = [torch.jit.fork(self.critics[i], x[:, :, i, :].unsqueeze(2)) for i in range(self.num_electrodes)]
#         scores = [torch.jit.wait(f) for f in futures]

#         # Inject electrode-specific embeddings into the output
#         scores = torch.stack(scores, dim=1)  # [B, E, 128]
#         scores += self.electrode_embeddings.unsqueeze(0)  # [1, E, 128]
#         scores = scores.view(scores.shape[0], -1)
#         # Fully connected layer for final prediction
#         scores = self.fc(scores)
#         return scores
    


# class Critic(nn.Module):
#     def __init__(self, num_electrodes=19, input_size=500, aggregation="sum", pretrained_critic=None):
#         super().__init__()
#         self.num_electrodes = num_electrodes
#         self.aggregation = aggregation  
#         self.input_size = input_size

#         self.critics = nn.ModuleList([
#             single_channel_critic(input_size) for _ in range(num_electrodes)
#         ])


#         if pretrained_critic is not None:
#             pretrained_state_dict = pretrained_critic.state_dict()
#             for critic in self.critics:
#                 critic.load_state_dict(pretrained_state_dict)

#             for critic in self.critics:
#                 for param in critic.parameters():
#                     param.data += 0.1*torch.randn_like(param)



#             reference_state_dict = self.critics[0].state_dict()


#             all_same = all(
#                 all(torch.equal(param, reference_state_dict[name]) for name, param in critic.state_dict().items())
#                 for critic in self.critics
#             )

#             print("All critics are identical:", all_same)

#         for critic in self.critics:
#             critic.is_single_channel = False



#         self.fc = nn.Sequential(
#             nn.Linear(128*num_electrodes, 1024),
#             nn.LeakyReLU(0.2), 
#             nn.LayerNorm([1024]),
#             nn.Linear(1024, 128),
#             nn.LeakyReLU(0.2), 
#             nn.LayerNorm([128]),
#             nn.Linear(128, 1),
#             )

#     def forward(self, x):
#         x = x.view(x.shape[0], 1, self.num_electrodes, self.input_size)
#         assert x.shape[2] == self.num_electrodes, "Mismatch in electrode count!"

#         futures = [torch.jit.fork(self.critics[i], x[:, :, i, :].unsqueeze(2)) for i in range(self.num_electrodes)]
#         scores = [torch.jit.wait(f) for f in futures]

#         scores = torch.cat(scores, dim=1)  
#         # scores = self.fc(scores)
#         return scores.mean(dim=1, keepdim=True) if self.aggregation == "mean" else scores.sum(dim=1, keepdim=True)


if __name__ == "__main__":
    sub_channel_test = torch.randn(32,  1, 3, 100)
    model = ElectrodeWiseConv(3, 1, 16, kernel_size=3, stride=2, padding=3//2)
    print(model(sub_channel_test).shape)

    latent_vector_size = 2000
    noise = torch.randn(32,  latent_vector_size)

    g = Generator(num_electrodes=19, latent_vector_size=latent_vector_size)
    print((out:=g(noise)).shape)
    c = Critic(num_electrodes=19)
    print((c(out)).shape)


