# import os
# import sys
# import numpy as np
# import shutil
# import torch.nn as nn
# import torch
# import argparse
# from scipy.signal import butter, filtfilt
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
# from modules import DCGAN

# def argparse_helper():
#     parser = argparse.ArgumentParser(description='Process some files.')
#     parser.add_argument('--model_paths', type=str, required=True, help='Path to the model (.pth) files, comma seperated.')
#     parser.add_argument('--save_locations', type=str, required=True, help='Path to the model (.pth) files, comma seperated.')
#     parser.add_argument('--num_2_gen', type=int, default=2000, help='Number of samples to generate.')

#     args = parser.parse_args() 
#     model_paths = args.model_paths.split(',') 
#     save_locations = args.save_locations.split(',') 
#     if len(model_paths) != len(save_locations):
#         raise ValueError("Expected the same number of model paths as save locations.")

#     return model_paths, save_locations, args.num_2_gen

# model_paths, save_locations, num_samples_to_generate = argparse_helper()


# for model_path, save_location in zip(model_paths,save_locations):
#     print(model_path)

#     #Normal Subjects
#     generator1 = torch.load(model_path)

#     # Assuming generator is your trained generator model
#     # and latent_vector_size is the size of the noise input for the generator

#     # Number of samples to generate

#     # Generate random latent vectors
#     noise = torch.randn(num_samples_to_generate, 100, device="cuda")

#     # Generating new data
#     generator1.eval()  # Switching the generator to evaluation mode

#     with torch.no_grad():  # No need to track gradients
#         generated_data1 = generator1(noise)
#         # Reshaping to 19x500 right after generation
#         generated_data1 = generated_data1.view(num_samples_to_generate, 19, -1)

#     # Converting to numpy for saving and further processing
#     generated_data1_np = generated_data1.cpu().numpy()
#     if not os.path.exists(save_location):
#         os.makedirs(os.path.dirname(save_location), exist_ok=True)

#     def minmax_scale_per_sample(data):
#         """Scales each sample in the dataset to range [-1, 1] independently."""
#         data_scaled = np.zeros_like(data)  # Create an array of the same shape
#         for i in range(data.shape[0]):  # Iterate over samples
#             min_val = data[i].min()
#             max_val = data[i].max()
#             data_scaled[i] = 2 * (data[i] - min_val) / (max_val - min_val) - 1 if max_val != min_val else data[i]
#         return data_scaled
#     def bandpass_filter(data, fs, lowcut=1.0, highcut=40.0, order=4):
#         """
#         Applies a zero-phase Butterworth bandpass filter to 3D EEG data.

#         Parameters:
#         - data (numpy array): EEG data, shape (samples, channels, time_steps)
#         - fs (float): Sampling frequency in Hz
#         - lowcut (float): Low cutoff frequency in Hz (default: 1.0 Hz)
#         - highcut (float): High cutoff frequency in Hz (default: 40.0 Hz)
#         - order (int): Order of the Butterworth filter (default: 4)

#         Returns:
#         - numpy array: Filtered EEG data with the same shape as input.
#         """
#         nyquist = 0.5 * fs  # Nyquist frequency
#         low = lowcut / nyquist
#         high = highcut / nyquist

#         # Design the bandpass filter
#         b, a = butter(order, [low, high], btype='band')

#         # Apply zero-phase filtering along the last axis (time dimension)
#         def filter_sample_channel(x):
#             return filtfilt(b, a, x, axis=-1)  # Apply along time dimension

#         # Apply filter across all samples and channels
#         filtered_data = np.apply_along_axis(filter_sample_channel, axis=-1, arr=data)

#         return filtered_data
#     generated_data1_np = np.asarray(minmax_scale_per_sample(generated_data1_np))
#     print(generated_data1_np.shape)
#     generated_data1_np = bandpass_filter(generated_data1_np, 256)


#     np.save(save_location, generated_data1_np)

# # #Patient Subjects
# # noise2 = torch.randn(num_samples_to_generate, 100, device="cuda")

# # generator2 = torch.load(model_path2)

# # with torch.no_grad():  # No need to track gradients
# #     generated_data2 = generator2(noise2)
# #     # Reshaping to 19x500 right after generation
# #     generated_data2 = generated_data2.view(num_samples_to_generate, 19, -1)

# # # Converting to numpy for saving and further processing
# # generated_data2_np = generated_data2.cpu().numpy()
# # if not os.path.exists(gen_data_save_location2):
# #     os.makedirs(os.path.dirname(gen_data_save_location2), exist_ok=True)

# # np.save(gen_data_save_location2, generated_data2_np)


import os
import sys
import numpy as np
import torch
import argparse
from scipy.signal import butter, filtfilt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def argparse_helper():
    parser = argparse.ArgumentParser(description='Generate and save EEG data.')
    parser.add_argument('--model_paths', type=str, required=True, help='Path to the model (.pth) files, comma separated.')
    parser.add_argument('--save_locations', type=str, required=True, help='Path to save the generated data, comma separated.')
    parser.add_argument('--num_2_gen', type=int, default=2000, help='Number of samples to generate per iteration.')
    parser.add_argument('--num_iterations', type=int, default=1, help='Number of times to generate and save data.')
    
    args = parser.parse_args()
    model_paths = args.model_paths.split(',')
    save_locations = args.save_locations.split(',')
    
    if len(model_paths) != len(save_locations):
        raise ValueError("Expected the same number of model paths as save locations.")
    
    return model_paths, save_locations, args.num_2_gen, args.num_iterations


def minmax_scale_per_sample(data):
    """Scales each sample in the dataset to range [-1, 1] independently."""
    data_scaled = np.zeros_like(data)
    print(data.shape[0])
    for i in range(data.shape[0]):
        min_val, max_val = data[i].min(), data[i].max()
        data_scaled[i] = 2 * (data[i] - min_val) / (max_val - min_val) - 1
    return data_scaled


def bandpass_filter(data, fs, lowcut=1.0, highcut=40.0, order=4):
    """Applies a zero-phase Butterworth bandpass filter to 3D EEG data."""
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return np.apply_along_axis(lambda x: filtfilt(b, a, x, axis=-1), axis=-1, arr=data)

from scipy.ndimage import gaussian_filter1d
def gaussian_smooth(data, sigma=2):
    """Applies a Gaussian smoothing filter along the last axis (time)."""
    return np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=sigma), axis=-1, arr=data)


model_paths, save_locations, num_samples_to_generate, num_iterations = argparse_helper()

for model_path, save_location in zip(model_paths, save_locations):
    generator = torch.load(model_path, map_location="cuda")
    generator.eval()
    os.makedirs(save_location, exist_ok=True)
    
    for i in range(num_iterations):
        noise = torch.randn(num_samples_to_generate, 8032, device="cuda")
        
        with torch.no_grad():
            generated_data = generator(noise).view(num_samples_to_generate, 19, -1)
        generated_data_np = generated_data.cpu().numpy()
        # generated_data_np = bandpass_filter(generated_data_np, fs=256)
        generated_data_np=gaussian_smooth(generated_data_np)
        generated_data_np = minmax_scale_per_sample(generated_data_np)
        print(generated_data_np[0].min())
        
        file_path = os.path.join(save_location, f'generated_data_{i}.npy')
        np.save(file_path, generated_data_np)
        print(f'Saved: {file_path}')
