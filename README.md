# EEGNet Inspired WGAN

## Extract .dat Files
* Take note of file path.
* Recommended location: ./Dataset/Unprocessed

## Run Preprocessing Script
Script only accepts .dat files. Split Control and Patient by adding a capital 'P'.

`python ./preprocessing.py --file_data_path <path to .dat files> --save_file_directory <Path to Dataset Folder> --seq_length <optional default=500> --skip <optional default=50>`

## Run WGAN_Training
Script exclusively loads .npy files. Data is of shape (samples, num_channels, seq_length)
`python ./WGAN_Training.py --file_data_path <path to .dat files>`

## Generate Data using model
Modify path variables in generate_data.ipynb. Run all.

## Validation
Modify path variables in Validation.ipynb and Validation.ipynb and run all. (These processes my take upwards of 1-3 hours depending on the script ran.)