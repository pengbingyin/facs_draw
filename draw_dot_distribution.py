import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # Set backend to TkAgg for interactive display
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
import os
import numpy as np

# Set global font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

# Define output_dir (adjust if needed based on previous setup)
output_dir = 'output/Z1A9-20ac'  # Replace with your actual output folder path

# List all .csv files in output_dir (and subdirs)
csv_files = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.lower().endswith('.csv'):
            csv_files.append(os.path.join(root, file))

if csv_files:
    for selected_csv in csv_files:
        # Get base name for saving
        base_name = os.path.splitext(os.path.basename(selected_csv))[0]
        print(f"Processing: {selected_csv}")

        # Load the CSV
        df = pd.read_csv(selected_csv)

        # Ensure columns exist
        required_cols = ['FSC-A', 'FITC-A', 'Pacific Blue-A']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Skipping {selected_csv}: Missing columns {missing_cols}")
            continue

        # Filter to positive values for log plotting (avoid log(0 or negative))
        df = df[(df['FSC-A'] > 0) & (df['FITC-A'] > 0) & (df['Pacific Blue-A'] > 0)]

        if df.empty:
            print(f"Skipping {selected_csv}: No positive data points")
            continue


        # Function to compute density for coloring
        def compute_density(x, y):
            if len(x) <= 20000:
                points = np.vstack([x, y])
                kde = gaussian_kde(points)
                density = kde.evaluate(points)
            else:
                bins = 151
                h, xedges, yedges = np.hist2d(x, y, bins=bins)
                xcenters = (xedges[:-1] + xedges[1:]) / 2
                ycenters = (yedges[:-1] + yedges[1:]) / 2
                density = interpn((xcenters, ycenters), h.T, np.array([x, y]).T, method='linear', bounds_error=False,
                                  fill_value=0)
            return density


        # Plot 1: FSC-A vs FITC-A with density coloring
        density_fitc = compute_density(df['FSC-A'], df['FITC-A'])
        fig1, ax1 = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
        scatter1 = ax1.scatter(df['FSC-A'], df['FITC-A'], s=0.1, c=density_fitc, cmap='viridis', marker='.')
        ax1.set_xlim(1, 1000000)
        ax1.set_ylim(1, 1000000)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('FSC-A')
        ax1.set_ylabel('FITC-A')
        fitc_save_path = os.path.join(output_dir, f'{base_name}_fitc.png')
        plt.savefig(fitc_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved FSC-A vs FITC-A plot to {fitc_save_path}")

        # Plot 2: FSC-A vs Pacific Blue-A with density coloring
        density_pb = compute_density(df['FSC-A'], df['Pacific Blue-A'])
        fig2, ax2 = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))
        scatter2 = ax2.scatter(df['FSC-A'], df['Pacific Blue-A'], s=0.1, c=density_pb, cmap='viridis', marker='.')
        ax2.set_xlim(1, 1000000)
        ax2.set_ylim(1, 1000000)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('FSC-A')
        ax2.set_ylabel('Pacific Blue-A')
        pb_save_path = os.path.join(output_dir, f'{base_name}_pb.png')
        plt.savefig(pb_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved FSC-A vs Pacific Blue-A plot to {pb_save_path}")

else:
    print("No CSV files found in output directory.")