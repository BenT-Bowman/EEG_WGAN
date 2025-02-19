import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from modules.DCGAN import Generator, Critic
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import argparse
torch.cuda.device(0)


"""
TODO: Devise model that learns one non-descript channel. Train on data.

Devise model that extends one-channel model with MoE type of layers. With each expert focused one specific channel, following the non-descript channel creation.
"""

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the input file. Data should be of .npy file type.')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to existing saved model.')
    parser.add_argument('--num_epochs', type=int, required=False, default=100, help='Number of epochs to train models for.')
    parser.add_argument('--gen_lr', type=float, required=False,    default=0.0001)
    parser.add_argument('--critic_lr', type=float, required=False, default=0.0001)
    parser.add_argument('--sleep', type=int, required=False, default=None, help='In case you don\'t want to cook your computer during long training loops.')
    parser.add_argument('--lambda_gp', '-l', type=int, required=False, default=10, help='Lambda, for GP')
    parser.add_argument('--critic_train_frequency', '-f', type=int, required=False, default=5, help='WGAN hyperparameter that controls how many times the critic is trained for how many times the generatoris trained.')
    parser.add_argument('--latent_vector_size', '-v', type=int, required=False, default=8032, help='Size of the latent vector for the generator.')
    args = parser.parse_args()

    return args.data_file_path, args.model_path, args.num_epochs, args.gen_lr, args.critic_lr, args.sleep, args.lambda_gp, args.critic_train_frequency, args.latent_vector_size

file_path, model_path, num_epochs, gen_lr, critic_lr, sleep_time, lambda_gp, freq, latent_vector_size = argparse_helper()
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
 
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



LAMBDA_GP = lambda_gp 
# num_epochs = 100
latent_vector_size = latent_vector_size 
critic_repeat = freq

##
# Training Prep
##

def gradient_penalty(critic, real_samples, fake_samples, device='cuda'):
    # Ensure real and fake samples are on the same device
    real_samples=real_samples.view(real_samples.size(0), 1, 19, 500)
    fake_samples=fake_samples.view(fake_samples.size(0), 1, 19, 500)
    real_samples = real_samples.to(device)
    fake_samples = fake_samples.to(device)

    # Generate random alpha values
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)

    # Interpolate between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Evaluate the critic on the interpolated samples
    d_interpolates = critic(interpolates)

    # Compute the gradients of the critic's output with respect to the interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten the gradients and compute the L2 norm
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

# generator = Generator(seq_length=500).to(device)
# critic =  Critic(seq_length=500).to(device)
if model_path is None:
    generator = Generator(latent_vector_size=latent_vector_size).to(device)
    critic =  Critic().to(device)
else:
    generator = torch.load(f'{model_path}/generator.pth')
    critic = torch.load(f'{model_path}/critic.pth')

d_lr = gen_lr # Llama suggests using lr < 10 times the original for fine tuning
g_lr = critic_lr # Llama suggests using lr < 10 times the original for fine tuning
optimizer_d = optim.Adam(critic.parameters(), lr=d_lr, betas=(0.0, 0.9))
optimizer_g = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))

##
# Training Loop
##
def track_gradients(model, epoch, batch_idx, layer_name=None):
    grad_norms = {}

    # Loop through model parameters and calculate gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()  # L2 norm of gradients
            grad_norms[name] = grad_norm

            # Print or log the gradient norms
            # if layer_name is None or layer_name in name:
            #     print(f"Epoch {epoch}, Batch {batch_idx}: Gradient Norm of {name}: {grad_norm}")

    return grad_norms

# Function to check for gradient vanishing/explosion
def check_gradient_explosion_or_vanishing(grad_norms, threshold_explode=1e3, threshold_vanish=1e-5):

    for layer, norm in grad_norms.items():
        if norm > threshold_explode:
            print(f"Gradient explosion detected in {layer} with norm {norm}")
        if norm < threshold_vanish:
            print(f"Gradient vanishing detected in {layer} with norm {norm}")

from tqdm import tqdm
from time import sleep
try:
    for epoch in range(num_epochs):
        avg_d=[]
        avg_g=[]
        avg_fake=[]
        avg_real=[]
        # print(epoch)
        pbar = tqdm(train_loader)
        for batch_idx, (real_samples, *_) in enumerate(pbar):
            real_samples = real_samples.to(device) #.view(real_samples.size(0), 1, 19, 500)
            # real_samples = real_samples.view(real_samples.size(0), 1, 19, 500)[:, :, 0, :].view(real_samples.size(0), -1)
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

                mean_real_preds = torch.mean(real_preds.detach())
                mean_fake_preds = torch.mean(fake_preds.detach())
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

            avg_fake.append(mean_fake_preds.cpu().numpy())
            avg_real.append(mean_real_preds.cpu().numpy())
            pbar.set_description(f"Epoch {epoch} Gen loss: {g_loss.item()} Critic loss: {d_loss.item()} ")
        print(f"\033[31mEpoch {epoch} Gen loss: {sum(avg_g)/len(avg_g)} Critic loss: {sum(avg_d)/len(avg_d)}\
               Average Fake Critic {sum(avg_fake)/len(avg_fake)} Average Real Critic {sum(avg_real)/len(avg_real)} \033[0m")
        if sleep_time is not None:
            print(f"Sleeping for {sleep_time}")
            sleep(sleep_time)
finally:
    import os
    print(file_path)
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