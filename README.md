# EEGNet Inspired WGAN

## Extract .dat Files
* Take note of file path.
* Recommended location: ./Dataset/Unprocessed

## Run Preprocessing Script
Script only accepts .dat files. Seperate Control and Patient by adding a capital 'P' in the first character of the patient string.

Usage: The script requires four command-line arguments:

* data_file_path: The path to the input file. The current data parsing capability is limited.
* save_file_directory: The directory to save the output file.
* skip: The number of lines to skip (default is 50).
* seq_length: The sequence length (default is 500).
Example Command:


`python process_files.py --data_file_path path/to/input_file.npy --save_file_directory path/to/save_directory --skip 100 --seq_length 600`\
This command will:

Process the file located at path/to/input_file.npy. \
Save the output to the directory path/to/save_directory. \
Skip the first 100 lines of the input file. \
Use a sequence length of 600.

## Run WGAN_Training

Usage: The script requires one command-line argument:

* data_file_path: The path to the input file. The file should be of .npy type.
Example Command:

`python process_data_file.py --data_file_path path/to/input_file.npy` \
This command will process the file located at path/to/input_file.npy.
### Notes
* Ensure the path provided in --data_file_path is valid and points to a .npy file.
## Generate Data using model

Usage: The script requires three command-line arguments:

* model_paths: A comma-separated list of paths to the model files (with .pth extension).
* save_locations: A comma-separated list of paths where the generated samples will be saved.
* num_2_gen (optional): The number of samples to generate. Defaults to 2000 if not provided.
Example Command:

`python process_models.py --model_paths path/to/model1.pth,path/to/model2.pth --save_locations path/to/save1,path/to/save2 --num_2_gen 1000`
This command will:

Load the models from path/to/model1.pth and path/to/model2.pth. \
Save the generated samples to path/to/save1 and path/to/save2. \
Generate 1000 samples. 

## Validation
Modify path variables in Validation.ipynb and MLProject.ipynb and run all. (These processes my take upwards of 1-3 hours depending on the script ran.)
