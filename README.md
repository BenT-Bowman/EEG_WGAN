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

--data_file_path: Required. Path to the input file in .npy format.
--model_path: Optional. Path to an existing saved model. Default is None.
--num_epochs: Optional. Number of epochs to train models for. Default is 100.
--gen_lr and --critic_lr: Optional. Learning rates for the generator and critic, respectively. Default is 0.0001.
--sleep: Optional. Time in seconds to pause the script after each epoch. Default is None.
To run the script, simply provide the required --data_file_path argument and any optional arguments you'd like to customize. For example:

``` python script.py --data_file_path path/to/data.npy --num_epochs 200 --gen_lr 0.001```
Note that the script will automatically use default values for any arguments that are not provided.
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

1. **Initial Training:** Train the model on MDD control data for 25 epochs with a learning rate of 1e-4.
2. **OCD Patient and Normal Data Training (Iteration 1):** Train two separate models on the OCD patient and normal data sets using the model created in step 1. Train for 100 iterations with a learning rate of 1e-5.
3. **OCD Patient and Normal Data Training (Iteration 2):** Train the two models created in step 2 again on the OCD patient and normal data sets, this time with a learning rate of 1e-6 and training for 100 iterations. \
\
This sequence of training configurations allows for a gradual refinement of the models, starting with a broad exploration of the MDD control data and then focusing on the OCD patient and normal data sets with increasingly fine-tuned parameters.
# Results
## *Generated Data* 
[![Generated](pictures\generated.png)](generated.png)
## *Real Data*
[![Generated](pictures\real.png)](real.png)

## ML Results

### Higuchi Fractal Dimension
[![Generated](pictures\hfd_result.png)](hfd_result.png)
### Entropy
[![Generated](pictures\entropy_result.png)](entropy_result.png)
### Kats Fractal Dimension
[![Generated](pictures\kfd_result.png)](kfd_result.png)
### EEGNET Classification
The model was trained on generated data and validated on real data. After training for 25 epochs, it achieved 100% accuracy.\
[![Generated](pictures\EEGNET_conf_mat.png)](EEGNET_conf_mat.png)
### EEGNET Discriminator
This model was trained to disciminatet between real and generated samples, it hovers around 60-65% accuracy in this task.\
[![Generated](pictures\EEGNET_discriminator.png)](EEGNET_discriminator.png)


