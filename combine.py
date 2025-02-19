import numpy as np
import os

# Path to the folder containing the .npy files
folder_path = "temp\Patient"

# List all .npy files in the folder
npy_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]

# Initialize an empty list to store arrays
arrays = []

# Load each .npy file and append to the list
for file in npy_files:
    array = np.load(file)
    arrays.append(array)

# Stack arrays along the first dimension
combined_array = np.concatenate(arrays, axis=0)

# Save the combined array to a new .npy file
output_file = "combined_array.npy"
np.save(output_file, combined_array)

print(f"Combined array saved to {output_file}.")
