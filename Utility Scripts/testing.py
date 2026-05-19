import os
import numpy as np
from tqdm import tqdm
import argparse

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the input file. Current data parsing capability limited.')
    args = parser.parse_args()

    return args.data_file_path

def main():
    file_path = argparse_helper()
    # List all files in the directory
    file_list = os.listdir(file_path)

    # Filter DAT files
    dat_files = [file for file in file_list if file.endswith('.dat')]

    # Display the list of DAT files
    if not dat_files:
        print("No DAT files found in the directory.")
        return
    else:
        print("DAT files in the directory:")
        for file in dat_files:
            print(file)

    patient = []
    control = []
    for file in dat_files:
        if file.startswith('P'):
            patient.append(np.loadtxt(os.path.join(file_path, file)))
        else:
            control.append(np.loadtxt(os.path.join(file_path, file)))

    patient = np.array(patient)[:, :, 250*20:250*20*2]
    control = np.array(control)[:, :, 250*20:250*20*2]
    print(control.max())
    # p_max_loc = np.unravel_index(np.argmax(patient), patient.shape)
    x, y, z = np.unravel_index(np.argmax(control), control.shape)
    control[x, y, z] = (control[x, y, z+1]+control[x, y, z-1])/2
    print(control.max())
    
    # print(patient[p_max], control[c_max])

    # print(patient.shape, control.shape)

    # patient_flat = patient.flatten()
    # control_flat = control.flatten()
    # patient_flat.sort()
    # control_flat.sort()
    # print(patient_flat)
    # print(control_flat)


if __name__ == "__main__":
    main()