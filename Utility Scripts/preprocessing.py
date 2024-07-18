import os
import numpy as np
from tqdm import tqdm
import argparse

##
# Utility Functions
##
def sliding_window(full_signal: np.ndarray, window: int = 500, features: int = 19, original_length: int = 40000, skip: int = 250):
    # print(full_signal.shape)
    assert full_signal.shape == (features, original_length)
    for idx in range(0, len(full_signal[-1]), skip):
        signal = full_signal[:, idx:idx+window]
        # print(signal.shape)
        if signal.shape != (features, window):
            break
        yield signal

def full(data: np.ndarray, **kwargs) -> np.ndarray: # TODO: Use only second 20 second window in full dataset. Data should be 200, 19, 500
    temp = []
    for signal in tqdm(data):
        for result_window in sliding_window(signal, **kwargs):
            temp.append(result_window)

    return np.asarray(temp)

def scale(data: np.ndarray)->np.ndarray:
    data_array = []
    compare = 100
    for signal in tqdm(data):
        if signal.max() > compare or abs(signal.min()) > compare:
            continue
        data_array.append(signal)
    data_array = np.asarray(data_array)
    print(data_array.mean(), data_array.std(), data_array.shape)
    return data_array

def scale_neg_one(data: np.ndarray)->np.ndarray:
    """Rescale data """
    data = data - data.min()
    return (data / data.max()) * 2 -1

def classification_full(data: np.ndarray, skip_first, window_size, **kwargs):
    skip_first = skip_first * 250
    window_size = window_size * 250
    # use_data: np.ndarray = data[skip_first:skip_first+window_size]
    temp = []
    # print(np.asarray(data).shape)
    # print(use_data)
    # print(skip_first, skip_first+window_size)
    for signal in tqdm(data):
        signal = signal[:, skip_first:skip_first+window_size]
        for result_window in sliding_window(signal, original_length=window_size, **kwargs):
            temp.append(result_window)
    # print(temp.shape)
    return np.asarray(temp)

##
# Main
##

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the input file. Current data parsing capability limited.')
    parser.add_argument('--save_file_directory', type=str, required=True, help='Directory to save the output file')
    parser.add_argument('--gen_skip', type=int, default=50, help='Number of lines to skip (default: 50)')
    parser.add_argument('--skip', type=int, default=20, help='Number of Seconds to skip')
    parser.add_argument('--seq_length', type=int, default=500, help='Sequence length (default: 500)')
    parser.add_argument('--usable_window', type=int, default=20, help='Number of Seconds to use of full dataset')
    args = parser.parse_args()

    return args.data_file_path, args.save_file_directory, args.gen_skip, args.seq_length, args.skip, args.usable_window

def main():
    file_path, save_file_directory, gen_skip, seq_length, real_skip, usable_window = argparse_helper()
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

    # control = full(control, skip=gen_skip, window=seq_length)
    # patient = full(patient, skip=gen_skip, window=seq_length)
    patient = classification_full(data=patient, skip_first=real_skip, window_size=usable_window, skip=gen_skip, window=seq_length)
    control = classification_full(data=control, skip_first=real_skip, window_size=usable_window, skip=gen_skip, window=seq_length)

    patient = np.asarray(patient)
    control = np.asarray(control)
    control = scale_neg_one(scale(control))
    patient = scale_neg_one(scale(patient))
    np.save(fr'{save_file_directory}/patient',patient )
    np.save(fr'{save_file_directory}/normal',control )
    print("Finished: ", patient.shape, control.shape)



    print("Classification")
    patient=[]
    control=[]
    for file in dat_files:
        if file.startswith('P'):
            patient.append(np.loadtxt(os.path.join(file_path, file)))
        else:
            control.append(np.loadtxt(os.path.join(file_path, file)))
    # print(patient)
    patient = classification_full(data=patient,skip_first=real_skip, window_size=usable_window, skip=seq_length, window=seq_length)
    control = classification_full(data=control,skip_first=real_skip, window_size=usable_window, skip=seq_length, window=seq_length)
    print("Finished: ", patient.shape, control.shape)
    np.save(fr'{save_file_directory}/patient_cls',patient )
    np.save(fr'{save_file_directory}/normal_cls',control )




if __name__ == "__main__":
    main()