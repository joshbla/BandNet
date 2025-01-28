import os
import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import eigvals

# Constants

neuro1 = 5000
neuro2 = int(neuro1 / 2)
samples = 500
training_range = [0.001, 1]
q_list = np.linspace(training_range[0], training_range[1], samples)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BandNet(nn.Module):
    def __init__(self, num_inputs):
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

def dispersion_curve(inputs, m, k_prime):
    global q_list


    ms = inputs[:m]
    kns = inputs[m:]

    if m == 1:
        argument = (2/ms[0]) * (np.sum(kns) - np.sum([kn * np.cos(math.pi * (n + 1) * q_list) for n, kn in enumerate(kns)], axis=0))
        argument = np.where(argument >= 0, argument, np.nan)
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
    
def calculate_metrics(target, design):
    # Find the maximum value from the design tensor
    max_value = torch.max(design)
    
    # Calculate the metrics using the original curves
    l1_distance = torch.mean(torch.abs(target - design))
    rmse = torch.sqrt(torch.mean(torch.pow(target - design, 2)))
    
    # Normalize the metrics by dividing by the maximum value
    normalized_l1_distance = l1_distance / max_value
    normalized_rmse = rmse / max_value
    
    return normalized_rmse.item(), normalized_l1_distance.item()

def main(data_folder, run_folder):
    model_folder = os.path.join(band_folder, run_folder)

    try:
        # Load the memory-mapped array
        output_file_path = os.path.join(data_folder, "outputs.npy")
        outputs = np.lib.format.open_memmap(output_file_path, mode='r')
    except FileNotFoundError:
        print(f"Memory-mapped file not found: {output_file_path}")
        sys.exit(1)

    try:
        # Load model
        model_path = os.path.join(model_folder, "model.pth")
        model = BandNet(num_inputs=num_inputs).to(device)  # Passes num_inputs to BandNet
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    total_rmse, total_l1 = 0, 0
    total_inputs = outputs.shape[0]  # Define the total number of inputs
    log_interval = total_inputs / 10  # Example: Log every 10% of the process
    next_log_point = log_interval

    for i in range(total_inputs):
        output_tensor = torch.tensor(outputs[i], device=device, dtype=torch.float32)  # Convert to tensor and transfer to the correct device
        
        design_input = model(output_tensor.unsqueeze(0)).detach().cpu().numpy()  # Model's predicted design inputs

        # Ensure all values before the mth element that are 0 or less to a small positive value
        design_input[:m] = np.where(design_input[:m] <= 0, 1e-6, design_input[:m])

        # Set negative values after mth position to 0
        design_input[m:] = np.where(design_input[m:] < 0, 0, design_input[m:])

        # Preprocess the designed input for dispersion_curve
        design_input_processed = np.insert(design_input, 0, 1)  # Add 1 for m1 at start
        design_input_processed = np.insert(design_input_processed, m, 1)  # Add 1 for k1 at m-th position
        
        # Generate designed outputs using the dispersion_curve function
        design_output = dispersion_curve(design_input_processed, m, k_prime)
        design_output = np.real(design_output)

        # Check for NaNs in design_output
        if np.isnan(design_output).any():
            print(f"NaN detected in designed output for output index {i}.")
            print(f"design_input to NaN: {design_input}")
            print(f"design_input_processed to NaN: {design_input_processed}")
            sys.exit()  # Skip further processing
        
        for j in range(m):
            design_output_tensor = torch.tensor(design_output[:, j + 1], dtype=torch.float32)
            target_output_tensor = torch.tensor(outputs[i][:, j + 1], dtype=torch.float32)

            # Calculate metrics
            rmse, l1 = calculate_metrics(target_output_tensor, design_output_tensor)
            total_rmse += rmse
            total_l1 += l1

        if i >= next_log_point:
            print(f'Processed {(i / total_inputs) * 100:.0f}% of inputs')
            next_log_point += log_interval


    num_outputs = outputs.shape[0]
    averaged_rmse = total_rmse / num_outputs
    averaged_l1 = total_l1 / num_outputs

    metadata = {
        "averaged_RMSE": averaged_rmse,
        "averaged_L1": averaged_l1,
    }

    json_path = os.path.join(data_folder, f"{run_folder}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Evaluation complete.")
    print("Results saved:", json_path)

if __name__ == "__main__":
    m = 1
    k = 5
    training_data = "1000"
    epochs = 100

    if len(sys.argv) < 5:
        print(f"\nUsage: python inference_set.py <m> <k> <data> <epochs>")
        print(f"Defaulting to m = {m}, k = {k}, data = {training_data}000, epochs = {epochs}\n")
    else:
        m, k, training_data, epochs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
        print(f"\nm = {m}, k = {k}, data = {training_data}, epochs = {epochs}\n")

    if k == 1:
        num_inputs = k + m - 1
    else:
        num_inputs = k + m - 2
    k_zeros = 0 if (k % 3 == 0) or (m != 3) else 3 - k % 3
    k_prime = k + k_zeros

    band_folder = f"{m}Band"
    training_data = training_data[:-3]
    run_folder = f"k-{k} training-{training_data}k epochs-{epochs}"
    data_folder = os.path.join(band_folder, "inference")

    main(data_folder, run_folder)