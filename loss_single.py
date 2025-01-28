import os
import json
import warnings
import numpy as np
import itertools
import matplotlib.pyplot as plt
from Core import dispersion_curve

# ===== USER CONFIGURATION =====
m = 1                     # Number of masses
target_input = np.array([1.0, 1.0, 4.0, 5.0, 9.0, 5.0])  # Format: [m1, m2..., k1, k2...]
design_input = np.array([1.0, 1.0, 3.999668598175049, 5.0051398277282715, 9.00297737121582, 5.005508899688721])
training_range = [0, 1]
samples = 500
# ==============================

q_list = np.linspace(training_range[0], training_range[1], samples)

# Calculate k and k_prime correctly
k = len(target_input[m:])  # Get k from the number of spring constants
k_zeros = 0 if (k % 3 == 0) or (m != 3) else 3 - k % 3
k_prime = k + k_zeros

def calculate_metrics(target, design):
    return (np.sqrt(np.mean((target - design)**2)), 
            np.mean(np.abs(target - design)))

def graphing(folder, target_result, design_result):
    os.makedirs(folder, exist_ok=True)
    
    lw = 8
    res = 120
    font = 35
    
    plt.figure(figsize=(9, 9), dpi=res)
    
    # Define colors for the first three pairs and then use default colors
    colors = itertools.cycle(['dodgerblue', 'r', 'g'] + [None]*(m-3))
    
    # Create legend elements outside the loop
    plt.plot([], [], 'k-', label='$\omega_i$ Target', linewidth=lw)
    plt.plot([], [], 'k--', label='$\omega_i$ Design', linewidth=lw)
    
    # Using a for-loop to create lines with distinct colors
    for i in range(1, m + 1):
        color = next(colors)
        plt.plot(q_list, target_result[:, -i], linestyle='-', color=color, linewidth=lw)
        plt.plot(q_list, design_result[:, -i], linestyle='--', color='k', linewidth=lw)
    
    # Add vertical lines at training range
    if training_range != [0, 1]:
        plt.axvline(x=training_range[0], color='g', linestyle='--', linewidth=lw*0.6, label=f'q={training_range[0]}')
        plt.axvline(x=training_range[1], color='g', linestyle='--', linewidth=lw*0.6, label=f'q={training_range[1]}')
    
    plt.legend(fontsize=font, loc='lower right')
    plt.xlabel("q", fontsize=font)
    plt.ylabel("ω", fontsize=font)
    plt.grid(True)
    plt.xticks(fontsize=font*.8)
    plt.yticks(fontsize=font*.8)
    
    # Set the limits for the first quadrant
    plt.xlim(0, max(q_list))
    
    # Find maximum ylim value
    overall_max = 0
    for i in range(1, m + 1):
        current_max = max(max(target_result[:, -i]), max(design_result[:, -i]))
        if current_max > overall_max:
            overall_max = current_max
    plt.ylim(0, overall_max * 1.1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    
    picture_path = os.path.join(folder, "plot test.png")
    plt.savefig(picture_path, dpi=res)
    plt.close()
    print(f"Test plot has been saved to {picture_path}")

# Add zeros if needed (for m=3 case)
if k_zeros > 0:
    target_input = np.append(target_input, np.zeros(k_zeros))
    design_input = np.append(design_input, np.zeros(k_zeros))

# Calculate curves using imported dispersion_curve function
target_curve = np.real(dispersion_curve(target_input, m, k_prime))
design_curve = np.real(dispersion_curve(design_input, m, k_prime))

# Calculate metrics
metrics = [calculate_metrics(target_curve[:, -i], design_curve[:, -i]) 
          for i in range(1, m+1)]
total_rmse, total_l1 = np.sum(metrics, axis=0)

# Save results
os.makedirs("temporary", exist_ok=True)
with open("temporary/metadata.json", "w") as f:
    json.dump({
        "test_loss_RMSE": total_rmse,
        "test_loss_L1": total_l1,
        "target_inputs": target_input.tolist(),
        "design_inputs": design_input.tolist(),
    }, f, indent=4)

# Generate plot using original graphing function
graphing("temporary", target_curve, design_curve)
print(f"Total RMSE: {total_rmse}")
print(f"Total L1 Loss: {total_l1}")