import warnings
import numpy as np
import itertools
import matplotlib.pyplot as plt
from Core import dispersion_curve
import os
import json
import re
import pandas as pd

class DispersionAnalyzer:
    def __init__(self, m, target_input, design_input, samples=500, training_range=[0, 1]):
        """
        Initialize the analyzer with input parameters.
        
        Args:
            m (int): Number of masses
            target_input (np.ndarray): Target input parameters [m1, m2..., k1, k2...]
            design_input (np.ndarray): Design input parameters [m1, m2..., k1, k2...]
            samples (int, optional): Number of samples. Defaults to 500.
            training_range (list, optional): Range for q. Defaults to [0, 1].
        """
        self.m = m
        self.target_input = np.array(target_input)
        self.design_input = np.array(design_input)
        self.samples = samples
        self.training_range = training_range
        self.q_list = np.linspace(training_range[0], training_range[1], samples)
        
        # Calculate k and k_prime
        self.k = len(self.target_input[m:])
        self.k_zeros = 0 if (self.k % 3 == 0) or (m != 3) else 3 - self.k % 3
        self.k_prime = self.k + self.k_zeros
        
        # Initialize curves
        self.target_curve = None
        self.design_curve = None

    def calculate_metrics(self, target, design):
        """Calculate RMSE, L1, and NRMSE between target and design curves."""
        l1 = np.mean(np.abs(target - design))
        rmse = np.sqrt(np.mean((target - design)**2))
        
        y_min, y_max = np.min(target), np.max(target)
        nrmse = rmse / (y_max - y_min) if (y_max - y_min) != 0 else 0
        
        return rmse, l1, nrmse

    def generate_curves(self):
        """Generate the dispersion curves for both target and design inputs."""
        # Add zeros if needed (for m=3 case)
        if self.k_zeros > 0:
            target_input = np.append(self.target_input, np.zeros(self.k_zeros))
            design_input = np.append(self.design_input, np.zeros(self.k_zeros))
        else:
            target_input = self.target_input
            design_input = self.design_input

        self.target_curve = np.real(dispersion_curve(target_input, self.m, self.k_prime))
        self.design_curve = np.real(dispersion_curve(design_input, self.m, self.k_prime))

    def compute_losses(self):
        """Compute all loss metrics between target and design curves."""
        if self.target_curve is None or self.design_curve is None:
            self.generate_curves()

        metrics = [self.calculate_metrics(self.target_curve[:, -i], self.design_curve[:, -i]) 
                  for i in range(1, self.m + 1)]
        return {
            "RMSE": np.sum([m[0] for m in metrics]),
            "L1": np.sum([m[1] for m in metrics]),
            "NRMSE": np.sum([m[2] for m in metrics])
        }

    def debug_plot(self, save_path="temporary/plot_test.png"):
        """Generate comparison plot for debugging purposes."""
        if self.target_curve is None or self.design_curve is None:
            self.generate_curves()

        lw = 8
        res = 120
        font = 35
        
        plt.figure(figsize=(9, 9), dpi=res)
        colors = itertools.cycle(['dodgerblue', 'r', 'g'] + [None]*(self.m-3))
        
        plt.plot([], [], 'k-', label='$\omega_i$ Target', linewidth=lw)
        plt.plot([], [], 'k--', label='$\omega_i$ Design', linewidth=lw)
        
        for i in range(1, self.m + 1):
            color = next(colors)
            plt.plot(self.q_list, self.target_curve[:, -i], linestyle='-', color=color, linewidth=lw)
            plt.plot(self.q_list, self.design_curve[:, -i], linestyle='--', color='k', linewidth=lw)
        
        if self.training_range != [0, 1]:
            plt.axvline(x=self.training_range[0], color='g', linestyle='--', 
                       linewidth=lw*0.6, label=f'q={self.training_range[0]}')
            plt.axvline(x=self.training_range[1], color='g', linestyle='--',
                       linewidth=lw*0.6, label=f'q={self.training_range[1]}')
        
        plt.legend(fontsize=font, loc='lower right')
        plt.xlabel("q", fontsize=font)
        plt.ylabel("ω", fontsize=font)
        plt.grid(True)
        plt.xticks(fontsize=font*.8)
        plt.yticks(fontsize=font*.8)
        
        plt.xlim(0, max(self.q_list))
        overall_max = max(np.max(self.target_curve[:, -i]) for i in range(1, self.m + 1))
        plt.ylim(0, overall_max * 1.1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        plt.savefig(save_path, dpi=res)
        plt.close()

def analyze_dispersion(m, target_input, design_input, debug=False):
    """
    Convenience function to perform analysis in one line.
    
    Args:
        m (int): Number of masses
        target_input (array-like): Target input parameters
        design_input (array-like): Design input parameters
        debug (bool, optional): Whether to generate debug plot. Defaults to False.
    
    Returns:
        dict: Dictionary containing the computed metrics
    """
    analyzer = DispersionAnalyzer(m, target_input, design_input)
    metrics = analyzer.compute_losses()
    
    if debug:
        analyzer.debug_plot()
    
    return metrics

if __name__ == "__main__":
    def parse_readable_string(readable_str):
        """Parse the readable string format into ms and ks arrays."""
        ms_match = re.search(r'ms = \[(.*?)\]', readable_str)
        ks_match = re.search(r'ks = \[(.*?)\]', readable_str)
        
        ms = [float(x.strip()) for x in ms_match.group(1).split(',')]
        ks = [float(x.strip()) for x in ks_match.group(1).split(',')]
        
        return np.array(ms + ks), ms, ks  # Return full array and separate ms, ks

    # Process all samples
    samples_dir = "samples"
    sample_types = ["samples_natural", "samples_stratified", "samples_uniform"]
    
    results_data = []
    
    for sample_type in sample_types:
        type_dir = os.path.join(samples_dir, sample_type)
        if not os.path.exists(type_dir):
            continue
            
        for sample_folder in os.listdir(type_dir):
            sample_path = os.path.join(type_dir, sample_folder)
            metadata_path = os.path.join(sample_path, "metadata.json")
            
            if not os.path.exists(metadata_path):
                continue
                
            # Read and parse metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Parse readable strings into input arrays
            target_input, target_ms, target_ks = parse_readable_string(metadata['readable_target'])
            design_input, design_ms, design_ks = parse_readable_string(metadata['readable_design'])
            
            # Get m and k
            m = len(target_ms)
            k = len(target_ks)
            
            # Analyze dispersion
            metrics = analyze_dispersion(m, target_input, design_input, debug=False)
            
            # Create a row of data
            row_data = {
                'Sample Type': sample_type,
                'Sample Name': sample_folder,
                'm': m,
                'k': k,
                'Target ms': str(target_ms),
                'Target ks': str(target_ks),
                'Design ms': str(design_ms),
                'Design ks': str(design_ks),
                'RMSE': metrics['RMSE'],
                'L1': metrics['L1'],
                'NRMSE': metrics['NRMSE'],
                'Target Readable': metadata['readable_target'],
                'Design Readable': metadata['readable_design']
            }
            
            results_data.append(row_data)
    
    # Create DataFrame and export to Excel
    df = pd.DataFrame(results_data)
    
    # Define column order
    column_order = [
        'Sample Type', 
        'Sample Name', 
        'm', 
        'k',
        'RMSE', 
        'L1', 
        'NRMSE',
        'Target ms',
        'Target ks',
        'Design ms',
        'Design ks',
        'Target Readable',
        'Design Readable'
    ]
    
    # Reorder and export
    df = df[column_order]
    excel_path = os.path.join(samples_dir, 'samples_analysis.xlsx')
    df.to_excel(excel_path, index=False)
    
    print(f"Analysis complete. Results saved to '{excel_path}'")
