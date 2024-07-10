import os
import sys
import numpy as np
import shutil
import torch.nn as nn
import torch
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from modules import DCGAN

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--model_paths', type=str, required=True, help='Path to the model (.pth) files, comma seperated.')
    parser.add_argument('--save_locations', type=str, required=True, help='Path to the model (.pth) files, comma seperated.')
    parser.add_argument('--num_2_gen', type=int, default=2000, help='Number of samples to generate.')

    args = parser.parse_args() 
    model_paths = args.model_paths.split(',') 
    save_locations = args.save_locations.split(',') 
    if len(model_paths) != len(save_locations):
        raise ValueError("Expected the same number of model paths as save locations.")

    return model_paths, save_locations, args.num_2_gen

model_paths, save_locations, num_samples_to_generate = argparse_helper()


for model_path, save_location in zip(model_paths,save_locations):
    print(model_path)

    #Normal Subjects
    generator1 = torch.load(model_path)

    # Assuming generator is your trained generator model
    # and latent_vector_size is the size of the noise input for the generator

    # Number of samples to generate

    # Generate random latent vectors
    noise = torch.randn(num_samples_to_generate, 100, device="cuda")

    # Generating new data
    generator1.eval()  # Switching the generator to evaluation mode

    with torch.no_grad():  # No need to track gradients
        generated_data1 = generator1(noise)
        # Reshaping to 19x500 right after generation
        generated_data1 = generated_data1.view(num_samples_to_generate, 19, -1)

    # Converting to numpy for saving and further processing
    generated_data1_np = generated_data1.cpu().numpy()
    if not os.path.exists(save_location):
        os.makedirs(os.path.dirname(save_location), exist_ok=True)

    np.save(save_location, generated_data1_np)


# #Patient Subjects
# noise2 = torch.randn(num_samples_to_generate, 100, device="cuda")

# generator2 = torch.load(model_path2)

# with torch.no_grad():  # No need to track gradients
#     generated_data2 = generator2(noise2)
#     # Reshaping to 19x500 right after generation
#     generated_data2 = generated_data2.view(num_samples_to_generate, 19, -1)

# # Converting to numpy for saving and further processing
# generated_data2_np = generated_data2.cpu().numpy()
# if not os.path.exists(gen_data_save_location2):
#     os.makedirs(os.path.dirname(gen_data_save_location2), exist_ok=True)

# np.save(gen_data_save_location2, generated_data2_np)