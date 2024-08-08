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


`python process_files.py --data_file_path path/to/input_file.npy --save_file_directory path/to/save_directory --skip 100 --seq_length 500`\
This command will:

Process the file located at path/to/input_file.npy. \
Save the output to the directory path/to/save_directory. \
Skip the first 100 lines of the input file. \
Use a sequence length of 500.

## Run WGAN_Training

### Required Arguments

* data_file_path: Path to the input file. Data should be of .npy file type.
### Optional Arguments

* model_path: Path to an existing saved model. If not provided, a new model will be trained from scratch.
* num_epochs: Number of epochs to train models for. Default is 100.
* gen_lr: Learning rate for the generator. Default is 0.0001.
* critic_lr: Learning rate for the critic. Default is 0.0001.
* sleep: Time (in seconds) to pause the script after each epoch. If not provided, the script will run continuously. 


Example 

To train a model on a file named data.npy and save it to model.h5, run the script with the following command:
\
`python WGAN_Training.py --data_file_path data.npy --model_path model.h5` \
You can customize the training process by adding additional arguments. For example, to train for 200 epochs with a learning rate of 0.001: 

`python WGAN_Training.py --data_file_path data.npy --num_epochs 200 --gen_lr 0.00001`
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

## Process
Training Configurations

1. **Initial Training:** Train the model on MDD control data for 10 epochs with a learning rate of 1e-4.
2. **OCD Patient and Normal Data Training (Iteration 1):** Train two separate models on the OCD patient and normal data sets using the model created in step 1. Train for 1000 iterations with a learning rate of 1e-5. Note: Signal snippets with any value over 100 were removed from the Generator training dataset.
