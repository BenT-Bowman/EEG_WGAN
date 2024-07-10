import os
import sys
import numpy as np
import shutil
import torch.nn as nn
import torch
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # This is hellish, I am not sure why relative imports did not work
from modules.DCGAN import Generator
# Path to the folder you want to empty
model_path1 = r'..\saved_models\test\generator_normal.pth'
gen_data_save_location1 = r'..\Dataset\Generated\gen_normal.npy'
real_data_path1 = r'..\Dataset\Real\normal.npy'

model_path2 = r'..\saved_models\test\generator_patient.pth'
gen_data_save_location2 = r'..\Dataset\Generated\gen_patient.npy'
real_data_path2 = r'..\Dataset\Real\patient.npy'

num_samples_to_generate = 2000

#Normal Subjects
generator1 = torch.load(model_path1)

# Assuming generator is your trained generator model
# and latent_vector_size is the size of the noise input for the generator

# Number of samples to generate

# Generate random latent vectors
noise1 = torch.randn(num_samples_to_generate, 100, device="cuda")

# Generating new data
generator1.eval()  # Switching the generator to evaluation mode

with torch.no_grad():  # No need to track gradients
    generated_data1 = generator1(noise1)
    # Reshaping to 19x500 right after generation
    generated_data1 = generated_data1.view(num_samples_to_generate, 19, -1)

# Converting to numpy for saving and further processing
generated_data1_np = generated_data1.cpu().numpy()
if not os.path.exists(gen_data_save_location1):
    os.makedirs(os.path.dirname(gen_data_save_location1), exist_ok=True)

np.save(gen_data_save_location1, generated_data1_np)


#Patient Subjects
noise2 = torch.randn(num_samples_to_generate, 100, device="cuda")

generator2 = torch.load(model_path2)

with torch.no_grad():  # No need to track gradients
    generated_data2 = generator2(noise2)
    # Reshaping to 19x500 right after generation
    generated_data2 = generated_data2.view(num_samples_to_generate, 19, -1)

# Converting to numpy for saving and further processing
generated_data2_np = generated_data2.cpu().numpy()
if not os.path.exists(gen_data_save_location2):
    os.makedirs(os.path.dirname(gen_data_save_location2), exist_ok=True)

np.save(gen_data_save_location2, generated_data2_np)