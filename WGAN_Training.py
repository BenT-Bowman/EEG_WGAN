import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from modules.DCGAN import Generator, Critic
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import argparse

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the input file. Data should be of .npy file type.')
    args = parser.parse_args()

    return args.data_file_path

file_path = argparse_helper()
device = "cuda" if torch.cuda.is_available() else "cpu"

##
# Load data
##

data = np.load(file_path)
print(f"{data.mean()=}, {data.std()=}, {data.shape=}")

data_tensor = torch.Tensor(data)

##
# Prepping Dataset
##

data_tensor = data_tensor.view(data_tensor.size(0), -1)  
 
dataset = TensorDataset(data_tensor)
 
batch_size = 128  
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



LAMBDA_GP = 10 
num_epochs = 2_000
latent_vector_size = 100 
critic_repeat = 5

##
# Training Prep
##

def gradient_penalty(critic, real_samples, fake_samples, device='cuda'):
    alpha = torch.rand_like(real_samples, device=device)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

generator = Generator(seq_length=500).to(device)
critic =  Critic(seq_length=500).to(device)

d_lr = 0.0001
g_lr = 0.0001
optimizer_d = optim.Adam(critic.parameters(), lr=d_lr, betas=(0.0, 0.9))
optimizer_g = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))

##
# Training Loop
##

from tqdm import tqdm
try:
    for epoch in range(num_epochs):
        avg_d=[]
        avg_g=[]
        # print(epoch)
        pbar = tqdm(train_loader)
        for (real_samples, *_) in pbar:
            real_samples = real_samples.to(device) #.view(real_samples.size(0), 1, 19, 500)
            # real_samples = real_samples.view(real_samples.size(0), 19, -1)
            critic.train()
            generator.eval()
            for _ in range(critic_repeat):
                # Fake samples
                noise = torch.randn((real_samples.size(0), latent_vector_size), device=device)
                fake_samples = generator(noise)
                
                real_preds = critic(real_samples)
                fake_preds = critic(fake_samples.detach())

                gp = gradient_penalty(critic, real_samples, fake_samples, device)
                
                d_loss = torch.mean(fake_preds) - torch.mean(real_preds) + LAMBDA_GP * gp
                # Backpropagation and optimization
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                # for p in critic.parameters():
                #     p.data.clamp_(-0.01, 0.01)
            critic.eval()
            generator.train()
            noise = torch.randn(batch_size, latent_vector_size, device=device)
            fake_samples = generator(noise)

            outputs = critic(fake_samples)

            g_loss = -torch.mean(outputs)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            avg_d.append(d_loss.item())
            avg_g.append(g_loss.item())
            pbar.set_description(f"Epoch {epoch} Gen loss: {g_loss.item()} Critic loss: {d_loss.item()} ")
        print(f"\033[31mEpoch {epoch} Gen loss: {sum(avg_g)/len(avg_g)} Critic loss: {sum(avg_d)/len(avg_d)} \033[0m")
finally:
    import os
    while True:
        file_directory = input("Where to save? ")
        try:
            if not os.path.exists(file_directory):
                os.makedirs(file_directory)
            break
        except Exception as e:
            print(e)
            continue

    torch.save(critic, fr"{file_directory}\critic.pth")
    torch.save(generator, fr"{file_directory}\generator.pth")
    print("Finished")