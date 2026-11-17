# Dependencies: python3 -m venv venv; source venv/bin/activate; pip install numpy matplotlib torch
import os
import csv
import math
import time
import json
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, TensorDataset, random_split

# Constants
m = 1  # Number of m values
k = 5  # Number of kns
training_data = 1000
grid = 500
epochs = 3
neuro1 = 5000
neuro2 = int(neuro1 / 2)
samples = int(math.sqrt(grid))
batch_multiplier = 8
batches = batch_multiplier * 128
learning_rate = batch_multiplier * k / 5 * 1e-8 * 750
generate = True
save = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_folder = f"1Band_2D/k-{k} training-{int(training_data / 1000)}k epochs-{epochs}"

# Functions for Data Generation and Processing
def dispersion_curve(kns, q_values_x, q_values_y):

    FN_values = [-2 * kn for kn in kns]
    FN_values[0] = 2 * sum(kns)

    sum_x = sum([FN * np.cos(n * q_values_x * math.pi) for n, FN in enumerate(FN_values)])
    sum_y = sum([FN * np.cos(n * q_values_y * math.pi) for n, FN in enumerate(FN_values)])

    argument = sum_x + sum_y

    ω = np.sqrt(argument)

    return ω

def band_points(inputs):
    q_x = np.linspace(0.001, 1, samples)
    q_y = np.linspace(0.001, 1, samples)
    q_values_grid = np.array(np.meshgrid(q_x, q_y)).T.reshape(-1, 2)
    ω_values = []
    for qx, qy in q_values_grid:
        ω = dispersion_curve(inputs, qx, qy)
        ω_values.append((qx, qy, ω))
    return np.array(ω_values)

def generate_points(input_values):
    result = band_points(input_values)
    if np.isnan(result).any():
        print(f"Result shape: {np.array(result).shape}")
        with open('problematic_inputs.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(input_values)
    return result

def generate_data(testtype, data):

    if testtype == "stratified":
        # Unified input for m values and kns
        inputs = stratified_inputs(data)
    elif testtype == "uniform":
        inputs = uniform_inputs(data)

    inputs = np.random.rand(data, k) * 10
    inputs[:, 0] = 1  # Setting all k1 values to 1

    start_time = time.time()
    with multiprocessing.Pool() as pool:
        sample_values = pool.map(generate_points, inputs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nGeneration Time: {elapsed_time} sec")

    sample_array = np.array(sample_values)
    nan_count = np.isnan(sample_array).sum()
    total_elements = np.prod(sample_array.shape)
    nan_percentage = (nan_count / total_elements) * 100
    if np.isnan(sample_values).any():
        print(f"There are {nan_count} NaN values in the sample_values array.")
        print(f"{nan_percentage:.2f}% of the values in the sample_values array are NaN.")
        assert False, "NaN value detected in sample_values."

    return sample_array, inputs

def stratified_inputs(data):
    # Unified input for kns
    inputs = np.random.rand(data, k) * 10

    if k != 1:
        # Setting the ratio-dependent k value to 1
        inputs[:, 0] = 1

    for row in inputs:
        # Randomly decide not to add zeros, or add at least one zero, making no zeros scenario twice as common
        choice = np.random.choice(['no_zeros', 'add_zeros'], p=[1/2, 1/2])
        if choice == 'add_zeros':
            # Deciding how many values to set to zero - at least one
            num_to_zero = np.random.randint(1, inputs.shape[1])
            # Generating unique random positions to set to zero
            zero_positions = np.random.choice(inputs.shape[1], size=num_to_zero, replace=False)
            row[zero_positions] = 0

    return inputs

def uniform_inputs(data):
    # Unified input for m values and kns
    inputs = np.random.rand(data, k) * 10

    if k != 1:
        # Setting the ratio-dependent k1 value to 1
        inputs[:, 0] = 1

    return inputs

# Model Definition
class BandNet(nn.Module):
    def __init__(self):
        super(BandNet, self).__init__()
        self.fc1 = nn.Linear(samples**2 * 3, neuro1)
        self.fc2 = nn.Linear(neuro1, neuro2)
        self.fc3 = nn.Linear(neuro2, neuro1)
        self.fc4 = nn.Linear(neuro1, neuro2)
        self.fc5 = nn.Linear(neuro2, neuro1)
        self.fc6 = nn.Linear(neuro1, k)

    def forward(self, x):
        x = x.view(-1, samples**2 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def save_model(model, filename="model.pth"):
    model_path = os.path.join(model_folder, filename)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"Model has been saved as '{model_path}'")


# Main method to encapsulate the script logic
def main():
    if generate:
        start_time = time.time()
        sample_data, inputs_data = generate_data("stratified", training_data)
        end_time = time.time()
        generation_time = end_time - start_time
        if save:
            np.savez('data.npz', X=sample_data, y=inputs_data)
        X = torch.Tensor(sample_data)
        y = torch.Tensor(inputs_data)
    else:
        data = np.load('data.npz')
        X = torch.Tensor(data['X'])
        y = torch.Tensor(data['y'])


    train_dataset = TensorDataset(X, y)

    sample_data, inputs_data = generate_data("stratified", training_data // 20)
    X = torch.Tensor(sample_data)
    y = torch.Tensor(inputs_data)

    test_dataset_stratified = TensorDataset(X, y)
    sample_stratified = test_dataset_stratified[0]

    sample_data, inputs_data = generate_data("uniform", training_data // 20)
    X = torch.Tensor(sample_data)
    y = torch.Tensor(inputs_data)

    test_dataset_uniform = TensorDataset(X, y)
    sample_uniform = test_dataset_uniform[0]

    num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 4
    trainloader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers)
    testloader_stratified = DataLoader(test_dataset_stratified, batch_size=batches, shuffle=False, num_workers=num_workers)
    testloader_uniform = DataLoader(test_dataset_uniform, batch_size=batches, shuffle=False, num_workers=num_workers)


    model = BandNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, threshold=0.0001)

    training_time, epoch_losses, lr_changes = training_loop(trainloader, model, criterion, optimizer, scheduler)

    print("Testing Stratified")
    stratified_RMSE, stratified_L1 = testing_loop(testloader_stratified, model, criterion, sample_stratified, f"Testing - Stratified")
    print("Testing Uniform")
    uniform_RMSE, uniform_L1 = testing_loop(testloader_uniform, model, criterion, sample_uniform, f"Testing - Uniform")

    save_model(model)

    # Save the metadata
    json_path = os.path.join(model_folder, "metadata.json")
    metadata = {
        "generation_time": generation_time,
        "training_time": training_time,
        "stratified_RMSE": stratified_RMSE,
        "stratified_L1": stratified_L1,
        "uniform_RMSE": uniform_RMSE,
        "uniform_L1": uniform_L1,
        "epoch_losses": epoch_losses,
        "lr_changes": lr_changes
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata has been saved")

def graph_results(target_input, design_input, filename):
    q_x = np.linspace(0.001, 1, grid)
    q_y = np.linspace(0.001, 1, grid)
    QX, QY = np.meshgrid(q_x, q_y)

    target_ω = dispersion_curve(target_input, QX, QY)
    design_ω = dispersion_curve(design_input, QX, QY)

    plot_heatmap(design_ω, 'Design ω', f"{filename}_heatmap_design.png", q_x, q_y)
    plot_3d_contour(target_ω, design_ω, f"{filename}_3d_contour.png", q_x, q_y)

    #plot_IBZ(target_ω, 'Target ω', f"{filename}_curve_target.png", q_x, q_y)
    #plot_IBZ(design_ω, 'Design ω', f"{filename}_curve_design.png", q_x, q_y)

def plot_heatmap(ω, label, filename, q_x, q_y):
    lw = 5
    res = 120
    font = 35

    plt.figure(figsize=(16, 9), dpi=res)
    img = plt.imshow(ω, extent=[q_x.min(), q_x.max(), q_y.min(), q_y.max()], origin='lower', aspect='auto', cmap='jet')
    cbar = plt.colorbar(img)
    cbar.set_label(label, size=font)
    cbar.ax.tick_params(labelsize=font * 0.8)
    plt.xlabel("$q_x$", fontsize=font)
    plt.ylabel("$q_y$", fontsize=font)
    plt.xticks(fontsize=font * 0.8)
    plt.yticks(fontsize=font * 0.8)
    plt.grid(True)

    # Using warnings to catch and ignore expected matplotlib warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    plt.savefig(os.path.join(model_folder, filename), dpi=res)

def plot_IBZ(ω, label, filename, q_x, q_y):
    lw = 5
    res = 120
    font = 35

    # Identify the key points in the IBZ you want the path to pass through. 
    symmetry_points = [
        # Replace 'start', 'middle', 'end' with the actual points in your IBZ.
        ('Start', (0,0)),
        ('Middle', (1,1)),
        ('End', (1,0.5)),
        ('Start', (0,0))  # back to the start point
    ]

    plt.figure(figsize=(16, 9), dpi=res)

    for i in range(len(symmetry_points) - 1):
        start_label, start = symmetry_points[i]
        end_label, end = symmetry_points[i+1]

        # Create an array of linearly interpolated points between the start and end points.
        n_points = 100  # You can adjust this to suit your needs
        xvals = np.linspace(start[0], end[0], n_points)
        yvals = np.linspace(start[1], end[1], n_points)

        # Interpolate the ω values along this line
        ω_along_line = np.array([ω[j, i] for i, j in zip(np.searchsorted(q_x, xvals), np.searchsorted(q_y, yvals))])

        # Plot this section of the dispersion curve
        plt.plot(np.linspace(i, i+1, n_points), ω_along_line)  # modify x-coordinate to be simply the section number

    plt.xlabel('Path through IBZ', fontsize=font)
    plt.ylabel(label, fontsize=font)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(model_folder, filename), dpi=res)

def plot_3d_contour(target_ω, design_ω, filename, q_x, q_y, levels=20):
    res = 120  # Resolution for saving the figure
    font = 35  # Font size for labels and titles
    lw = 10    # Linewidth for the contours

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(11, 9), dpi=res)

    QX, QY = np.meshgrid(q_x, q_y)

    # Design ω Contour: Draw this first so it is at the bottom of the Z-order
    ax.contour3D(QX, QY, design_ω, levels, cmap='Greys', linewidths=lw, linestyles='dashed')

    # Target ω Contour: Draw this after so it will be rendered on top by default
    ax.contour3D(QX, QY, target_ω, levels, cmap='jet', linewidths=lw)

    ax.set_xlabel('$q_x$', fontsize=font, labelpad=10)
    ax.set_ylabel('$q_y$', fontsize=font, labelpad=10)
    ax.set_zlabel('ω', fontsize=font, labelpad=10)
    ax.tick_params(labelsize=font * 0.5)

    # Create proxy artists for the legend
    proxy_target = Line2D([0], [0], linestyle='-', color='black', linewidth=lw - 5)
    proxy_design = Line2D([0], [0], linestyle='--', color='black', linewidth=lw - 5)

    ax.legend([proxy_target, proxy_design], ['Target ω', 'Design ω'], fontsize=(font - 10))

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(model_folder, filename), dpi=res)

def training_loop(trainloader, model, criterion, optimizer, scheduler):
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
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch + 1}/{epochs}]\t\tLoss: {avg_loss}")

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

    return training_time, epoch_losses, lr_changes


def testing_loop(testloader, model, criterion, sample, filename):
    total_loss = 0.0
    for points, labels in testloader:
        points, labels = points.to(device), labels.to(device)
        outputs = model(points)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    test_loss_RMSE = math.sqrt(total_loss / len(testloader))
    test_loss_L1 = total_loss / len(testloader)  # Assuming we're using L1Loss for this computation
    print(f"Test Set Loss (RMSE): {test_loss_RMSE}")
    print(f"Test Set Loss (L1): {test_loss_L1}")

    # Get the first test sample for graphing
    test_sample, target_input = sample
    design_input = model(test_sample.unsqueeze(0).to(device)).detach().cpu().numpy()[0]

    # Call graphing function
    graph_results(target_input.numpy(), design_input, filename)

    return test_loss_RMSE, test_loss_L1

if __name__ == "__main__":
    # If the folder exists, append a number to the name
    counter = 1
    temp_folder = model_folder
    while os.path.exists(temp_folder):
        temp_folder = f"{model_folder}({counter})"
        counter += 1
    model_folder = temp_folder
    os.makedirs(model_folder)

    main()