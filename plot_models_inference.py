import os
import re
import sys
import json
import matplotlib.pyplot as plt

# Capture m from command line arguments
if len(sys.argv) < 2:
    m = 1
    print(f"\nUsage: python plot_models_inference.py <m>")
    print(f"Defaulting to m = {m}\n")
else:
    m = int(sys.argv[1])

# Setup for file access
band_folder = f"{m}Band"
folder_path = os.path.join(band_folder, 'inference')

# Adjust regex for new filename pattern
file_pattern = re.compile(r'k-(\d+)')

# Initialize storage lists
rmse_values = []
l1_values = []
file_numbers = []

file_count = 0
for filename in os.listdir(folder_path):
    match = file_pattern.match(filename)
    if match:
        file_number = int(match.group(1))
        file_numbers.append(file_number)

        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)
            rmse_values.append(data.get('averaged_RMSE'))
            l1_values.append(data.get('averaged_L1'))
        file_count += 1

# Inform about processed files and errors
if file_count == 0:
    print("No matching files found. Please verify the folder_path and file_pattern.")
else:
    print(f"m = {m}")
    print(f"Processed {file_count} files.")
    if not rmse_values or not l1_values:
        print("rmse_values or l1_values list is empty. Please verify the JSON file contents.")
    else:
        # Calculate data bounds
        data_min = min(filter(None, rmse_values + l1_values))
        data_max = max(filter(None, rmse_values + l1_values))

        # Sort the data before plotting
        sorted_indices = sorted(range(len(file_numbers)), key=file_numbers.__getitem__)
        sorted_rmse_values = [rmse_values[i] for i in sorted_indices]
        sorted_l1_values = [l1_values[i] for i in sorted_indices]

        # Create tick labels
        tick_labels = [f"M{file_numbers[i]}" for i in sorted_indices]

        # Plotting settings for high resolution and quality
        res = 120  # Resolution for saving the figure
        font = 25  # Font size for labels and titles
        lw = 3     # Line width for plot lines

        plt.figure(figsize=(12, 12), dpi=res)
        plt.plot(sorted(file_numbers), sorted_rmse_values, label='Normalized RMSE', marker='o', linewidth=lw)
        plt.plot(sorted(file_numbers), sorted_l1_values, label='Normalized L1', marker='x', linewidth=lw)

        plt.xlabel('Model Complexity', fontsize=font)
        plt.ylabel('Percentage Error', fontsize=font)
        plt.title('Model Error on 20K Sets of M5 Testing Bands', fontsize=font)
        plt.ylim([data_min, data_max])
        plt.xticks(sorted(file_numbers), tick_labels, fontsize=font * 0.8)
        plt.yticks(fontsize=font * 0.8)
        plt.legend(fontsize=font * 0.8)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'plot.png'), bbox_inches='tight', dpi=res)
