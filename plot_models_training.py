# These should probably be bar graphs, not line graphs. But I don't have time to change it now.

import os
import re
import csv
import sys
import json
import numpy as np 
import matplotlib.pyplot as plt 


# Constants
m = 1
training_data = "100"
epochs = 100

if len(sys.argv) < 4:
    print(f"\nUsage: python plot_models_training.py <m> <data> <epochs>")
    print(f"Defaulting to m = {m}, data = {training_data}000, epochs = {epochs}\n")
else:
    m = sys.argv[1]
    training_data = sys.argv[2][:-3]
    epochs = sys.argv[3]
    print(f"m = {m}, data = {training_data}000, epochs = {epochs}")

band_folder = f"{m}Band"

# Extracting the range of x values dynamically from folder names within band_folder
folder_names = os.listdir(band_folder)
k_values = []
for folder_name in folder_names:
    match = re.match(r'k-(\d+)', folder_name)
    if match:
        k_values.append(int(match.group(1)))

if not k_values:
    print("No matching folders found.")
    sys.exit(1)

min_k, max_k = min(k_values), max(k_values)
x_values = range(min_k, max_k + 1)  # Adjusted to include max_k

csv_data = [["k", "Stratified RMSE", "Stratified L1", "Uniform RMSE", "Uniform L1"]]

# Loop through the x values to process each folder and collect data
for x in x_values:
    folder_name = f"k-{x} training-{training_data}k epochs-{epochs}"
    file_path = os.path.join(band_folder, folder_name, "metadata.json")

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            csv_data.append([
                x,
                data["stratified_RMSE"],
                data["stratified_L1"],
                data["uniform_RMSE"],
                data["uniform_L1"]
            ])
    except FileNotFoundError:
        print(f"Metadata file not found in folder: {folder_name}")
        continue

# Write the collected data to a CSV file
csv_file_path = f"{band_folder}/training/metrics_summary.csv"
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Data saved to {csv_file_path}")

# Extract lists for plotting, skipping the header row
_, stratified_RMSEs, stratified_L1s, uniform_RMSEs, uniform_L1s = zip(*csv_data[1:])

# Convert string lists to float for plotting
stratified_RMSEs = [float(i) for i in stratified_RMSEs]
stratified_L1s = [float(i) for i in stratified_L1s]
uniform_RMSEs = [float(i) for i in uniform_RMSEs]
uniform_L1s = [float(i) for i in uniform_L1s]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x_values, stratified_RMSEs, label='Stratified RMSE', marker='o')
plt.plot(x_values, stratified_L1s, label='Stratified L1', marker='o')
plt.plot(x_values, uniform_RMSEs, label='Uniform RMSE', marker='o')
plt.plot(x_values, uniform_L1s, label='Uniform L1', marker='o')

plt.title('Loss by Model Type')
plt.xlabel('Model trained to k non-local interactions')
plt.ylabel('Loss')
plt.legend()
plt.xticks(np.arange(min_k, max_k + 1, (max_k - min_k) // 10 or 1))
plt.grid(axis='x', which='both', linestyle='--', linewidth=0.7)
plt.xlim(min_k, max_k)

# Save the plot to a file
plt.savefig(f"{band_folder}/training/metrics_comparison.png")
print("Graph saved as metrics_comparison.png")