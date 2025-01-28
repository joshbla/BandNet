#python3 -m venv venv
#source venv/bin/activate
# or
#venv\Scripts\activate
#pip install numpy matplotlib torch psutil

import os
import gc
import sys
import math
import time
import json
import psutil
import logging
import warnings
import builtins
import threading
import numpy as np
from numpy.linalg import eigvals
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
global m, k, num_inputs, k_prime, k_zeros, samples, q_list, epochs, training_range, training_data, num_workers, device



# Parameters

args =                              [1, 5, 100000, 100]
training_range =                    [0.001, 1]
learning_multiplier =               1e-8 * 750
patience =                          4
test_ratio =                        20
old_epochs =                        100

# Flags

generate =                          True
train =                             True
test =                              True
show_plots =                        False
train_type =                        "stratified"
test_types =                        ["stratified", "uniform", "natural"]

# Constants

samples =                           500
neuro1 =                            5000
neuro2 =                            int(neuro1/2)
batch_multiplier =                  8
batches =                           batch_multiplier * 128
q_list =                            np.linspace(training_range[0], training_range[1], samples)
num_workers =                       4 * torch.cuda.device_count() if torch.cuda.is_available() else 4
device =                            torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():



    # Generate or Load Data

    if generate:
        print("Generating data...")

        outputs, inputs, generation_time = generate_data(train_type, training_data, data_folder)
        if outputs is None:
            print("Data generation failed. Exiting...")
            sys.exit(1)
            
        X = torch.Tensor(outputs)
        y = torch.Tensor(inputs)
    elif train:
        try:
            print("Loading data...")

            # Paths for memory-mapped files
            output_file_path = os.path.join(previous_data_folder, "outputs.npy")
            input_file_path = os.path.join(previous_data_folder, "inputs.npy")

            # Load the memory-mapped arrays
            outputs = np.lib.format.open_memmap(output_file_path, mode='r+')
            inputs = np.lib.format.open_memmap(input_file_path, mode='r+')

            print("Data has been loaded\n")
        except Exception as err:
            print(f"Error while loading data: {err}")
            sys.exit(1)

        X = torch.Tensor(outputs)
        y = torch.Tensor(inputs)

        # Load generation time from metadata.json
        metadata_path = os.path.join(band_folder, previous_name, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
                generation_time = metadata['generation_time']  # Retrieve the generation_time
        else:
            print("metadata.json file not found; generation time could not be loaded.")
            generation_time = 0  # Fallback if metadata.json is not found



    # Train or Load Model

    model = BandNet().to(device)

    if train:
        print("Training the model...")

        # Create DataLoader
        train_dataset = TensorDataset(X, y)
        trainloader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers)

        # Loss and Optimizer
        criterion = nn.L1Loss()  # Using Mean Absolute Error (MAE) instead of MSE
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=gamma, patience=patience, threshold=threshold)

        start_time = time.time()

        lr_changes = {0: learning_rate}
        epoch_losses = {}

        for epoch in range(epochs):
            running_loss = 0.0
            for points, labels in trainloader:
                points, labels = points.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(points)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss/len(trainloader)
            print(f"Epoch [{epoch+1}/{epochs}]\t\tLoss: {avg_loss}")

            # Record the current learning rate before update
            current_lr = optimizer.param_groups[0]['lr']

            # Update the learning rate based on the average training loss
            scheduler.step(avg_loss)

            # Access the learning rate after update to see if it has changed
            new_lr = scheduler.optimizer.param_groups[0]['lr']
            if current_lr != new_lr:
                print(f"Learning rate adjusted to {new_lr}")
                lr_changes[epoch+1] = new_lr

            # Store the average loss with the epoch number as key
            epoch_losses[epoch+1] = avg_loss

        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining Time: {training_time} sec\n")

        model_path = os.path.join(model_folder, "model.pth")
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        print(f"\nModel has been saved in:\n'{model_path}'\n")

        graph_training(model_folder, epoch_losses, lr_changes)
    else:
        try:
            print("Loading the model...")
            model_path = os.path.join(previous_model_folder, "model.pth")
            model.load_state_dict(torch.load(model_path))
            print("Model has been loaded\n")
        except Exception as err:
            print(f"Error while loading the model: {err}")
            sys.exit(1)

        # Load training time from metadata.json
        metadata_path = os.path.join(band_folder, previous_name, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)
                training_time = metadata['training_time']



    # Test the Model

    if test:
        print("Testing the model...")
        stratified_RMSE = stratified_L1 = uniform_RMSE = uniform_L1 = natural_RMSE = natural_L1 = 0
        for testtype in test_types:
            print("")
            print(f"{testtype.capitalize()} data")

            # Create test data folder as subfolder of data_folder
            test_data_folder = os.path.join(data_folder, f"test_{testtype}")
            os.makedirs(test_data_folder, exist_ok=True)

            outputs, inputs, _ = generate_data(testtype, training_data // test_ratio, test_data_folder)

            X = torch.Tensor(outputs)
            y = torch.Tensor(inputs)

            test_dataset = TensorDataset(X, y)
            base_picture_path = os.path.join(model_folder, f"plot_{testtype}_test.png")
            try:
                if testtype == "stratified":
                    stratified_RMSE, stratified_L1 = testing(model, test_dataset, base_picture_path, testtype)
                elif testtype == "uniform":
                    uniform_RMSE, uniform_L1 = testing(model, test_dataset, base_picture_path, testtype)
                elif testtype == "natural":
                    natural_RMSE, natural_L1 = testing(model, test_dataset, base_picture_path, testtype)
            except Exception as e:
                print(f"Error during {testtype} testing: {e}")



    # Saving

    # Save the test loss, target inputs, and design inputs as a json
    json_path = os.path.join(model_folder, "metadata.json")
    metadata = {
        "generation_time": generation_time,
        "training_time": training_time,
        "stratified_RMSE": stratified_RMSE,
        "stratified_L1": stratified_L1,
        "uniform_RMSE": uniform_RMSE,
        "uniform_L1": uniform_L1,
        "natural_RMSE": natural_RMSE,
        "natural_L1": natural_L1,
        "epoch_losses": epoch_losses,
        "lr_changes": lr_changes
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata has been saved")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    while True:
        memory_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # Memory in GB
        logging.info(f"Current Memory Usage: {memory_usage:.2f} GB")
        logging.info(f"Current Time: {time.strftime('%H:%M:%S', time.localtime())}")
        time.sleep(10)  # Log every 10 seconds

def log_print(*args, **kwargs):
    # Print to the console
    builtins.print(*args, **kwargs)
    # Log the same message
    logging.info(' '.join(map(str, args)))

def generate_data(testtype, data, save_folder):

    if testtype == "stratified":
        inputs = stratified_inputs(data)
    elif testtype == "uniform":
        inputs = uniform_inputs(data)
    elif testtype == "natural":
        inputs = natural_inputs(data)
    else:
        raise ValueError(f"Unknown test type: {testtype}")

    # Store original inputs for dispersion_curve
    original_inputs = inputs.copy()

    # Process inputs to final dimensions for saving and training
    if k > 1:
        # Removing the ratio-dependent k value
        inputs = np.delete(inputs, m, axis=1)
    # Removing the ratio-dependent m value
    inputs = np.delete(inputs, 0, axis=1)
    # Remove the n_zeros values from the end of each row
    if k_zeros > 0:
        inputs = inputs[:, :-k_zeros]

    # Save processed inputs
    input_file_path = os.path.join(save_folder, "inputs.npy")
    try:
        print("Saving inputs data...")
        memmap_inputs = np.lib.format.open_memmap(input_file_path, mode='w+', 
                                                dtype=inputs.dtype, 
                                                shape=inputs.shape)
        memmap_inputs[:] = inputs
        memmap_inputs.flush()
        if hasattr(memmap_inputs, '_mmap'):
            memmap_inputs._mmap.close()
        print("Inputs have been saved\n")
    except Exception as e:
        print(f"Error saving inputs file: {e}")
        logging.exception(f"Error saving inputs file: {e}")
        return None, None, 0

    start_time = time.time()
    total_inputs = len(inputs)

    # Get sample output using original input format
    sample_output = dispersion_curve(original_inputs[0], m, k)
    output_shape = (total_inputs,) + sample_output.shape
    
    # Create memory-mapped output file
    output_file = os.path.join(save_folder, "outputs.npy")
    outputs_mmap = None
    try:
        outputs_mmap = np.lib.format.open_memmap(output_file, 
                                               mode='w+',
                                               dtype=np.float64,
                                               shape=output_shape)
    except Exception as e:
        print(f"Failed to create memory-mapped file: {e}")
        logging.exception(f"Failed to create memory-mapped file: {e}")
        return None, None, 0

    chunk_size = data // 20  # Number of inputs to process in each chunk
    next_log_point = total_inputs * 0.05
    pool = None

    try:
        pool = Pool()
        # Break inputs into smaller chunks for processing
        for i in range(0, total_inputs, chunk_size):
            original_input_chunk = original_inputs[i : i + chunk_size]
            results_chunk = []

            # Check if we've processed more than 5% of total
            if i >= next_log_point:
                print(f'Processed {(i / total_inputs) * 100:.0f}% of inputs')
                next_log_point += total_inputs * 0.05

            for input in original_input_chunk:
                result = pool.apply_async(dispersion_curve, args=(input, m, k))
                results_chunk.append(result)

            # Write chunk results directly to memory-mapped file
            chunk_outputs = np.array([result.get() for result in results_chunk])
            
            # Validate chunk before writing to disk
            nan_count = np.isnan(chunk_outputs).sum()
            if nan_count > 0:
                chunk_size = len(chunk_outputs)
                nan_percentage = (nan_count / np.prod(chunk_outputs.shape)) * 100
                print(f"There are {nan_count} NaN values in chunk starting at index {i}")
                print(f"{nan_percentage:.2f}% of the values in this chunk are NaN")

                # Get the indices of the NaN values in outputs
                nan_indices = np.argwhere(np.isnan(chunk_outputs))
                # Print 5 inputs that resulted in NaN outputs, if available
                for idx in nan_indices[:5]:
                    print(f"Input that resulted in NaN output: {original_input_chunk[idx[0]]}")
                raise ValueError("NaN value detected in outputs.")

            outputs_mmap[i:i+len(chunk_outputs)] = chunk_outputs
            outputs_mmap.flush()  # Ensure this chunk is written to disk

            # Clear chunk data
            del chunk_outputs
            del results_chunk
            del original_input_chunk

        print('Processed 100% of inputs')
        pool.close()
        pool.join()

        # Clear original inputs as they're no longer needed
        del original_inputs
        gc.collect()

        # Ensure all data is written to disk
        outputs_mmap.flush()
        
        # Close the memory map
        if outputs_mmap is not None and hasattr(outputs_mmap, '_mmap'):
            outputs_mmap._mmap.close()

        print("Outputs have been completely saved to disk")

        # Load the saved data in read mode
        outputs = np.load(output_file, mmap_mode='r')

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        logging.exception(f"An error occurred during processing: {e}")
        # Don't delete the output file - it contains partial results
        if pool is not None:
            pool.close()
        return None, None, 0
    
    finally:
        # Ensure cleanup happens no matter what
        if outputs_mmap is not None and hasattr(outputs_mmap, '_mmap'):
            try:
                outputs_mmap._mmap.close()
            except Exception as e:
                logging.exception(f"Error during cleanup: {e}")

    end_time = time.time()
    generation_time = end_time - start_time

    # Now working with the loaded data
    try:
        print(f"\nGeneration Time: {generation_time} sec\n")

        imaginary_distribution(outputs)
        outputs = np.real(outputs)

        # Add memory checks around the critical line
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024 * 1024)  # Memory in GB
        logging.info(f"\nBefore final array conversion - Current Memory Usage: {memory_before:.2f} GB")
        logging.info("Converting memory-mapped array to regular numpy array (potential memory spike)...")
        
        outputs = np.array(outputs)
        
        memory_after = process.memory_info().rss / (1024 * 1024 * 1024)  # Memory in GB
        logging.info(f"After final array conversion - Current Memory Usage: {memory_after:.2f} GB")
        logging.info(f"Memory increase: {(memory_after - memory_before):.2f} GB\n")

        return outputs, inputs, generation_time

    except Exception as e:
        print(f"An error occurred during post-processing: {e}")
        logging.exception(f"An error occurred during post-processing: {e}")
        return None, None, 0

def stratified_inputs(data):
    # Unified input for m values and kns
    inputs = np.random.rand(data, num_inputs) * 10

    # Setting the ratio-dependent m value to 1
    inputs = np.insert(inputs, 0, 1, axis=1)  # Insert at 0th position

    if k != 1:
        # Setting the ratio-dependent k value to 1
        inputs = np.insert(inputs, m, 1, axis=1)  # Insert at mth position
  
    for row in inputs:
        # Randomly decide not to add zeros, or add at least one zero, making no zeros scenario twice as common
        choice = np.random.choice(['no_zeros', 'add_zeros'], p=[1/2, 1/2])
        if choice == 'add_zeros':
            # Deciding how many values to set to zero - at least one, up to the number of columns after m
            num_to_zero = np.random.randint(1, inputs.shape[1] - m + 1)
            # Generating unique random positions to set to zero, ensuring positions are after m
            zero_positions = np.random.choice(range(m, inputs.shape[1]), size=num_to_zero, replace=False)
            row[zero_positions] = 0

    # Append 0s at the last k_zeros positions
    zeros_to_append = np.zeros((data, k_zeros))
    inputs = np.hstack((inputs, zeros_to_append))

    return inputs

def uniform_inputs(data):
    # Unified input for m values and kns
    inputs = np.random.rand(data, num_inputs) * 10

    # Setting the ratio-dependent m value to 1
    inputs = np.insert(inputs, 0, 1, axis=1)  # Insert at 0th position

    if k != 1:
        # Setting the ratio-dependent k value to 1
        inputs = np.insert(inputs, m, 1, axis=1)  # Insert at mth position

    # Append 0s at the last k_zeros positions
    zeros_to_append = np.zeros((data, k_zeros))
    inputs = np.hstack((inputs, zeros_to_append))

    return inputs

def natural_inputs(data):
    # Generate random integers between 0 and 9 inclusive for all values
    inputs = np.random.randint(0, 10, size=(data, num_inputs))
    
    # For the first m values (excluding the ratio-dependent m value), 
    # generate random integers between 1 and 9
    inputs[:, :m-1] = np.random.randint(1, 10, size=(data, m-1))

    # Setting the ratio-dependent m value to 1
    inputs = np.insert(inputs, 0, 1, axis=1)  # Insert at 0th position

    if k != 1:
        # Setting the ratio-dependent k value to 1
        inputs = np.insert(inputs, m, 1, axis=1)  # Insert at mth position

    # Append 0s at the last k_zeros positions
    zeros_to_append = np.zeros((data, k_zeros))
    inputs = np.hstack((inputs, zeros_to_append))

    return inputs

def dispersion_curve(inputs, m, k_prime):
    global q_list


    ms = inputs[:m]
    kns = inputs[m:]

    if m == 1:
        argument = (2/ms[0]) * (np.sum(kns) - np.sum([kn * np.cos(math.pi * (n + 1) * q_list) for n, kn in enumerate(kns)], axis=0))
        # Applying a threshold to handle round-off errors close to zero
        argument = np.where(np.abs(argument) < 1e-10, 0, argument)
        ω = np.sqrt(argument)

        return np.stack([q_list, ω], axis=1)
    elif m == 2:
        cos_values_K0 = [kn * np.cos((math.pi/2) * n * q_list) if n % 2 == 0 else np.zeros_like(q_list) for n, kn in enumerate(kns, start=1)]
        K0 = np.sum(kns) - np.sum(cos_values_K0, axis=0)

        cos_values_K1 = [kn * np.cos((math.pi/2) * n * q_list) if n % 2 != 0 else np.zeros_like(q_list) for n, kn in enumerate(kns, start=1)]
        K1 = 2 * np.sum(cos_values_K1, axis=0)

        term1 = K0 * (1/ms[0] + 1/ms[1])
        term2 = np.sqrt((K0)**2 * (1/ms[0] + 1/ms[1])**2 + (1/(ms[0]*ms[1])) * (K1**2 - 4 * K0**2))

        ω_1 = term1 + term2
        ω_2 = term1 - term2

        ω_optical = np.sqrt(ω_1)   # (upper branch)
        ω_acoustic = np.sqrt(ω_2)  # (lower branch)

        result = np.stack([q_list, ω_optical, ω_acoustic], axis=1)
        return result
    elif m == 3:
        # Define the given values and constants
        M_indexes = int(k_prime/3)
        a = math.pi/3
        ω_list_combined = []

        for q in q_list:
            # The internal calculations remain largely the same;
            # we're only pushing the loop from the outer function to this inner function.

            A = -2 * np.sum(kns) + np.sum([kns[3*M - 1] * (np.exp(1j*q*3*M*a) + np.exp(-1j*q*3*M*a)) for M in range(M_indexes)])
            B = np.sum([kns[3*M - 3] * np.exp(1j*q*(3*M - 3)*a) + kns[3*M - 2] * np.exp(-1j*q*3*M*a) for M in range(M_indexes)])
            C = np.sum([kns[3*M - 3] * np.exp(-1j*q*3*M*a) + kns[3*M - 2] * np.exp(1j*q*3*(M - 1)*a) for M in range(M_indexes)])

            delta = np.exp(1j*q*3*a)

            # v = λ M_inv * K v
            M_inv_K = np.array([
                [-1/ms[0] * A       , -1/ms[0] * B      , -1/ms[0] * C  ],
                [-delta * C / ms[1] , -A / ms[1]        , -B / ms[1]    ],
                [-delta * B/ ms[2]  , -delta * C / ms[2], -A / ms[2]    ]
            ])

            # Solve the generalized eigenvalue problem
            eigenvalues = eigvals(M_inv_K)

            ω_list = np.sqrt(np.abs(eigenvalues))  # Ensure no negative due to numerical issues
            ω_sorted = np.sort(ω_list)

            ω_list_combined.append([q] + ω_sorted.tolist())

        return np.array(ω_list_combined)

def imaginary_distribution(outputs):

    # Extract the imaginary parts
    imag_parts = np.imag(outputs)

    # Check if there are any imaginary parts before proceeding.
    if np.all(imag_parts == 0):
        return

    # Get the absolute values
    abs_imag_parts = np.abs(imag_parts)

    # Calculate the orders of magnitude (ignoring zero values to prevent negative infinity)
    non_zero_abs_imag_parts = abs_imag_parts[abs_imag_parts > 0]
    orders_of_magnitude = np.floor(np.log10(non_zero_abs_imag_parts))

    # Get unique orders and their counts
    unique_orders, counts = np.unique(orders_of_magnitude, return_counts=True)

    # Calculate the percentages
    total_values = abs_imag_parts.size
    percentages = (counts / total_values) * 100

    # Print the results
    for order, percentage in zip(unique_orders, percentages):
        print(f"Order of magnitude: 10^{int(order)}, Percentage: {percentage:.2f}%")

    # Find the index of the maximum absolute imaginary part
    max_imag_index = np.unravel_index(np.argmax(abs_imag_parts), outputs.shape)

    # Get the eigenvalue with the maximum absolute imaginary part
    max_imag_eigenvalue = outputs[max_imag_index]

    if max_imag_eigenvalue != 0:
        print(f"The eigenvalue with the largest imaginary part is: {max_imag_eigenvalue}\n")

    # Print the largest imaginary part
    max_imag_part = np.max(abs_imag_parts)
    print(f"The largest imaginary part is: {max_imag_part}\n")

class BandNet(nn.Module):
    def __init__(self):
        super(BandNet, self).__init__()
        self.fc1 = nn.Linear(samples * (m + 1), neuro1)
        self.fc2 = nn.Linear(neuro1, neuro2)
        self.fc3 = nn.Linear(neuro2, neuro1)
        self.fc4 = nn.Linear(neuro1, neuro2)
        self.fc5 = nn.Linear(neuro2, neuro1)
        self.fc6 = nn.Linear(neuro1, num_inputs)

    def forward(self, x):
        x = x.view(-1, samples * (m + 1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def format_readable_vector(vector):
    # Add back ratio-dependent values
    full_vector = np.insert(vector, 0, 1)  # Add m1=1
    if k > 1:
        full_vector = np.insert(full_vector, m, 1)  # Add k1=1 if k>1
    
    # Split into ms and ks
    ms = full_vector[:m]
    ks = full_vector[m:]
    
    # Format into readable string
    return f"ms = {ms.tolist()} , ks = {ks.tolist()}"

def testing(model, test_dataset, base_picture_path, test_type):
    testloader = DataLoader(test_dataset, batch_size=batches, shuffle=False, num_workers=num_workers)

    # Loss is MSE but we'll change this to RMSE later
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for points, labels in testloader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    test_loss_RMSE = math.sqrt(total_loss/len(testloader))
    print(f"Test Set Loss (RMSE): {test_loss_RMSE}")

    # Loss is MSE but we'll change this to RMSE later
    criterion = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for points, labels in testloader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    test_loss_L1 = total_loss/len(testloader)
    print(f"Test Set Loss (L1): {test_loss_L1}")

    # Create test samples folder
    test_samples_folder = os.path.join(os.path.dirname(base_picture_path), f"samples_{test_type}")
    os.makedirs(test_samples_folder, exist_ok=True)

    # Test 10 samples
    for i in range(10):
        # Create folder for this sample
        sample_folder = os.path.join(test_samples_folder, f"sample_{i+1}")
        os.makedirs(sample_folder, exist_ok=True)

        # Get test sample and its target kns
        test_sample, target_input = test_dataset[i]

        # Get the model's design kns for this sample
        design_input = model(test_sample.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

        # Create metadata file for this sample
        metadata_path = os.path.join(sample_folder, "metadata.json")
        
        # Convert target_input and design_input to readable format
        readable_target = format_readable_vector(target_input.numpy())
        readable_design = format_readable_vector(design_input)

        sample_metadata = {
            "target": target_input.numpy().tolist(),
            "design": design_input.tolist(),
            "readable_target": readable_target,
            "readable_design": readable_design
        }
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f, indent=4)

        # Create plot for this sample
        picture_path = os.path.join(sample_folder, "plot.png")
        graph_testing(picture_path, target_input, design_input)
    print(f"{test_type.capitalize()} samples have been saved")

    return test_loss_RMSE, test_loss_L1

def graph_training(model_folder, epoch_losses, lr_changes):
    lw = 4
    res = 120
    font = 24

    # Loss plot
    try:
        # Convert epoch_losses to a list of tuples and sort by epoch number
        epoch_loss_items = sorted((int(epoch), loss) for epoch, loss in epoch_losses.items())
        epochs_list, losses_list = zip(*epoch_loss_items)

        # Create the plot
        plt.figure(figsize=(16, 9), dpi=res)
        plt.plot(epochs_list, losses_list, label='Loss', linewidth=lw)

        # Logarithmic scale for the y-axis
        plt.yscale('log')

        # Add lines where the learning rate changed
        for epoch, lr in lr_changes.items():
            if int(epoch) != 0:  # Skip the initial learning rate if it's recorded as epoch 0
                plt.axvline(x=int(epoch), color='grey', linestyle='--', linewidth=lw * 0.5, alpha=0.7)

        # Labeling the plot
        plt.xlabel('Epoch', fontsize=font)
        plt.ylabel('Loss', fontsize=font)
        plt.title('Loss over Epochs and Learning Rate Adjustments', fontsize=font * 1.2)
        plt.legend(fontsize=font)

        # Grid and ticks configuration
        plt.grid(True)
        ax = plt.gca()
        # Set the maximum number of x-axis and y-axis labels to around 10
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.xticks(fontsize=font * 0.8)
        plt.yticks(fontsize=font * 0.8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()

        # Save the figure to the folder for the current run
        loss_plot_path = os.path.join(model_folder, "plot loss.png")
        plt.savefig(loss_plot_path)
        print("Loss plot has been saved\n")
        plt.close()
    except Exception as e:
        print(f"Error during plotting loss graph: {e}\n")
        plt.close()

def graph_testing(picture_path, target_input, design_input):
    lw = 8
    res = 120
    font = 35

    target_input = target_input.numpy()

    # Add the ratio-dependent inputs
    target_input = np.insert(target_input, 0, 1)  # Add 1 for m1 at the start
    if k > 1:
        target_input = np.insert(target_input, m, 1)  # Add 1 for k1 at the m-th position

    design_input = np.insert(design_input, 0, 1)  # Add 1 for m1 at the start
    if k > 1:
        design_input = np.insert(design_input, m, 1)  # Add 1 for k1 at the m-th position

    # Add the zeros at the end
    target_input = np.append(target_input, np.zeros(k_zeros))
    design_input = np.append(design_input, np.zeros(k_zeros))

    # Compute the results
    # Assuming dispersion_curve is a predefined function
    target_result = dispersion_curve(target_input, m, k_prime)
    target_result = np.real(target_result)

    design_result = dispersion_curve(design_input, m, k_prime)
    design_result = np.real(design_result)

    q = np.linspace(0, 1, samples)

    try:
        # Plot the curves
        plt.figure(figsize=(9, 9), dpi=res)

        # Define colors for the first three pairs and then use default colors
        colors = itertools.cycle(['dodgerblue', 'r', 'g'] + [None]*(m-3))

        # Create legend elements outside the loop
        plt.plot([], [], 'k-', label='$\omega_i$ Target', linewidth=lw)
        plt.plot([], [], 'k--', label='$\omega_i$ Design', linewidth=lw)

        # Using a for-loop to create lines with distinct colors
        for i in range(1, m + 1):
            color = next(colors)
            plt.plot(q, target_result[:, -i], linestyle='-', color=color, linewidth=lw)
            plt.plot(q, design_result[:, -i], linestyle='--', color='k', linewidth=lw)

        # Add vertical lines at training range
        if training_range != [0.001, 1]:
            plt.axvline(x=training_range[0], color='g', linestyle='--', linewidth=lw*0.6, label=f'q={training_range[0]}')
            plt.axvline(x=training_range[1], color='g', linestyle='--', linewidth=lw*0.6, label=f'q={training_range[1]}')

        plt.legend(fontsize=font, loc='lower right')
        plt.xlabel("q", fontsize=font)
        plt.ylabel("ω", fontsize=font)
        plt.grid(True)
        plt.xticks(fontsize=font*.8)
        plt.yticks(fontsize=font*.8)

        # Set the limits for the first quadrant
        plt.xlim(0, max(q))

        # Using a for-loop instead of find the maximum ylim value manually
        overall_max = 0
        start = 0 if k == 1 else 1
        for i in range(start, m + 1):
            current_max = max(max(target_result[:, -i]), max(design_result[:, -i]))
            if current_max > overall_max:
                overall_max = current_max
        plt.ylim(0, overall_max * 1.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()

        # Save the figure
        plt.savefig(picture_path, dpi=res)
        if show_plots:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Error during plotting: {e}")
        logging.exception(f"Error during plotting: {e}")
        plt.close()

if __name__ == "__main__":
    m, k, training_data, epochs = map(int, args[:4])

    if len(sys.argv) < 5:
        print(f"\nUsage: python Core.py <m> <k> <data> <epochs>")
        print(f"Defaulting to m = {m}, k = {k}, data = {training_data}, epochs = {epochs}\n")
    else:
        args = sys.argv[1:]
        m, k, training_data, epochs = int(args[0]), int(args[1]), int(args[2]), int(args[3])
        if int(m) > 3:
            print("The maximum value for m is 3.")
            sys.exit(1)
        print(f"\nm = {m}, k = {k}, data = {training_data}, epochs = {epochs}\n")

    # Only force flags OFF if epochs=0, otherwise respect the original flag values
    if epochs == 0:
        test = False 
        train = False

    # Set threshold and gamma based on m
    if m == 1 or m == 2:
        threshold = 0.0001
        gamma = 0.5
    elif m == 3:
        threshold = 0.001
        gamma = 0.7

    if k == 1:
        num_inputs = k + m - 1
    else:
        num_inputs = k + m - 2

    k_zeros = 0 if (k % 3 == 0) or (m != 3) else 3 - k % 3
    k_prime = k + k_zeros
    learning_rate = batch_multiplier * k / 5 * learning_multiplier

    data_thousands = int(training_data / 1000)
    base_name = f"k-{k} training-{data_thousands}k epochs-"
    previous_name = f"{base_name}{old_epochs}"
    current_name = f"{base_name}{epochs}"
    band_folder = f"{m}Band"

    # Create the inference folder
    inference_folder = os.path.join(band_folder, "inference")
    os.makedirs(inference_folder, exist_ok=True)

    # Create the inference readme file
    inference_readme = os.path.join(inference_folder, "README.txt")
    with open(inference_readme, 'w') as f:
        f.write("Place an output.npy file in this folder to run inference_set.py. This will perform inference on that set of curves using your specified model. \n")
    
    # Create the training folder
    training_folder = os.path.join(band_folder, "training")
    os.makedirs(training_folder, exist_ok=True)

    # Create the training readme file
    training_readme = os.path.join(training_folder, "README.txt")
    with open(training_readme, 'w') as f:
        f.write("Results from plot_models_training.py will appear here.\n")

    # Create the model folder
    model_folder = os.path.join(band_folder, current_name)
    previous_model_folder = os.path.join(band_folder, previous_name)

    # If the folder exists, append a number to the name
    counter = 1
    original_name = current_name
    while os.path.exists(model_folder):
        current_name = f"{original_name}({counter})"
        model_folder = os.path.join(band_folder, current_name)  # Update model_folder here
        counter += 1

    os.makedirs(model_folder, exist_ok=True)

    # Create the data folder
    data_folder = os.path.join(model_folder, "data")
    previous_data_folder = os.path.join(previous_model_folder, "data")

    os.makedirs(data_folder, exist_ok=True)

    # Create the logs folder
    logs_folder = os.path.join(model_folder, 'logs')
    os.makedirs(logs_folder, exist_ok=True)
    # Log filename: MM-SS.log
    log_filename = time.strftime("%M-%S.log", time.localtime())
    log_file_path = os.path.join(logs_folder, log_filename)
    
    # Update the logging configuration to use the new log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    
    # Redefine the print function
    print = log_print
    try:
        logging.info("Script started")
        # Start the memory usage logging thread
        memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
        memory_thread.start()
        # Your main script logic
        main()
        logging.info("Script finished successfully")
    except Exception as e:
        logging.exception("An error occurred: ")
    finally:
        gc.collect()