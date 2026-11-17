import os
import sys
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

input = [1.0, 5.0, 8.0, 9.0, 3.3]
m = 1
k = 5
training_data = 1000
grid = 500
epochs = 3
samples = int(math.sqrt(grid))
neuro1 = 5000
neuro2 = int(neuro1 / 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_folder = f"1Band_2D/k-{k} training-{int(training_data / 1000)}k epochs-{epochs}"


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


# Functions for processing and visualization
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


def load_model(filename="model.pth"):
    model = BandNet().to(device)
    model_path = os.path.join(model_folder, filename)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from '{model_path}'")
    return model


def process_single_input(model, custom_input):
    custom_input = np.array(custom_input)
    sample_data = generate_points(custom_input)
    sample_tensor = torch.Tensor(sample_data).unsqueeze(0).to(device)
    design_input = model(sample_tensor).detach().cpu().numpy()[0]
    return custom_input, design_input


def plot_heatmap(ω, label, filename, q_x, q_y):
    lw = 5
    res = 120
    font = 35

    plt.figure(figsize=(16, 16), dpi=res)  # Adjusting figsize to keep the figure square
    img = plt.imshow(
        ω,
        extent=[q_x.min(), q_x.max(), q_y.min(), q_y.max()],
        origin='lower',
        aspect='equal',  # Setting aspect to 'equal' for a square plot
        cmap='jet'
    )
    cbar = plt.colorbar(img)
    cbar.set_label(label, size=font)
    cbar.ax.tick_params(labelsize=font * 0.8)
    plt.xlabel("$q_x$", fontsize=font)
    plt.ylabel("$q_y$", fontsize=font)
    plt.xticks(fontsize=font * 0.8)
    plt.yticks(fontsize=font * 0.8)
    plt.grid(True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    plt.savefig(os.path.join(f"{model_folder}/inference", filename), dpi=res)



def plot_3d_contour(target_ω, design_ω, filename, q_x, q_y, levels=20):
    res = 120
    font = 35
    lw = 10

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(11, 9), dpi=res)
    QX, QY = np.meshgrid(q_x, q_y)

    # Design ω Contour
    ax.contour3D(QX, QY, design_ω, levels, cmap='Greys', linewidths=lw, linestyles='dashed', alpha=0.5)

    # Target ω Contour
    ax.contour3D(QX, QY, target_ω, levels, cmap='jet', linewidths=lw, alpha=0.5)

    ax.set_xlabel('$q_x$', fontsize=font, labelpad=10)
    ax.set_ylabel('$q_y$', fontsize=font, labelpad=10)
    ax.set_zlabel('ω', fontsize=font, labelpad=10)
    ax.tick_params(labelsize=font * 0.5)

    # Create proxy artists for the legend
    proxy_target = Line2D([0], [0], linestyle='-', color='black', linewidth=lw - 5, alpha=0.5)
    proxy_design = Line2D([0], [0], linestyle='--', color='black', linewidth=lw - 5, alpha=0.5)

    ax.legend([proxy_target, proxy_design], ['Target ω', 'Design ω'], fontsize=font - 10)

    plt.tight_layout()
    plt.savefig(os.path.join(f"{model_folder}/inference", filename), dpi=res)



def graph_results(target_input, design_input, filename):
    q_x = np.linspace(0.001, 1, grid)
    q_y = np.linspace(0.001, 1, grid)
    QX, QY = np.meshgrid(q_x, q_y)

    target_ω = dispersion_curve(target_input, QX, QY)
    design_ω = dispersion_curve(design_input, QX, QY)

    plot_heatmap(design_ω, 'Design ω', f"{filename}_heatmap_design.png", q_x, q_y)
    plot_3d_contour(target_ω, design_ω, f"{filename}_3d_contour.png", q_x, q_y)


def plot_and_save_results(target_input, design_input, filename_prefix):
    graph_results(target_input, design_input, filename_prefix)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"\nUsage: python inference_single_2D.py <m> <k> <training_data> <epochs>")
        print(f"Defaulting to m = {m}, k = {k}, training_data = {training_data}, epochs = {epochs}\n")
    else:
        args = sys.argv[1:]
        m, k, training_data, epochs = int(args[0]), int(args[1]), int(args[2]), int(args[3])

        if int(m) > 3:
            print("The maximum value for m is 3.")
            sys.exit(1)

        model_folder = f"1Band_2D/k-{k} training-{int(training_data / 1000)}k epochs-{epochs}"

        print(f"\nm = {m}, k = {k}, training_data = {training_data}, epochs = {epochs}\n")

    # Load model
    model = load_model()

    # Process custom input
    target_input, design_input = process_single_input(model, input)

    # Ensure the output folder exists
    inference_folder = os.path.join(model_folder, "inference")
    if not os.path.exists(inference_folder):
        os.makedirs(inference_folder)

    # Plot and save results
    input_str = "_".join([str(i) for i in input])
    plot_and_save_results(target_input, design_input, f"{input_str}")