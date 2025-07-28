import fcsparser
import os
import json  # For optional metadata export
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # Set backend to TkAgg for interactive display
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

# Set global font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

# Define the fluorescence background reference values
FITC_background_reference = 0.002208235
Pacific_Blue_background_reference = 0.011865047

# Define folder name (from your previous example)
folder_name = "Z1A9-200ac"

# Replace with the full path to your folder containing FCS files
input_dir = f'data/{folder_name}'  # e.g., 'data/Q13B7-20ac'

# Replace with the full path where you want to save the CSV (and optional JSON metadata) files
output_dir = f'output/{folder_name}'  # e.g., 'output/Q13B7-20ac'

# Option to include metadata: Set to True to export metadata as a separate JSON file for each FCS
export_metadata = True

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to collect fluorescence data for combined CSV
fluorescence_data = {}

# List to collect data for stacked plots (filename, fitc_data, pb_data)
plot_data_list = []

# Loop through all files in the input directory (and optionally subdirectories)
for root, dirs, files in os.walk(input_dir):  # Use os.walk for subfolders; or os.listdir(root) for flat
    for filename in files:
        if filename.lower().endswith('.fcs'):
            input_path = os.path.join(root, filename)
            try:
                # Parse the FCS file
                meta, data = fcsparser.parse(input_path, reformat_meta=True)

                # Optional: Print preview for verification (comment out for large batches)
                print(f"Processing {filename} - Metadata keys:", list(meta.keys()))
                print(f"Data preview for {filename}:", data.head())

                # Calculate FITC fluorescence
                if FITC_background_reference == 0:
                    raise ValueError("FITC_background_reference cannot be zero to avoid division by zero.")
                data['FITC_fluorescence'] = (data['FITC-A'] / data['FSC-A']) / FITC_background_reference - 1

                # Calculate Pacific Blue fluorescence
                if Pacific_Blue_background_reference == 0:
                    raise ValueError("Pacific_Blue_background_reference cannot be zero to avoid division by zero.")
                data['Pacific_Blue_fluorescence'] = (data['Pacific Blue-A'] / data[
                    'FSC-A']) / Pacific_Blue_background_reference - 1

                # Store for combined CSV (using base filename without extension)
                base_name = os.path.splitext(filename)[0]
                fluorescence_data[f"{base_name}_FITC"] = data['FITC_fluorescence']
                fluorescence_data[f"{base_name}_PacificBlue"] = data['Pacific_Blue_fluorescence']

                # Collect data for plotting
                fitc_data = data['FITC_fluorescence'].dropna().values
                pb_data = data['Pacific_Blue_fluorescence'].dropna().values
                if len(fitc_data) > 1 and len(pb_data) > 1:
                    plot_data_list.append((filename, fitc_data, pb_data))
                else:
                    print(f"Not enough data points to plot densities for {filename}")

                # Generate output CSV filename and save the data (including new columns)
                relative_path = os.path.relpath(root, input_dir)  # Preserve subfolder structure if using os.walk
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                csv_name = os.path.splitext(filename)[0] + '.csv'
                output_path = os.path.join(output_subdir, csv_name)
                data.to_csv(output_path, index=False)

                print(f"Converted data from {filename} to {output_path}")

                # Optional: Export metadata to JSON
                if export_metadata:
                    json_name = os.path.splitext(filename)[0] + '_meta.json'
                    json_path = os.path.join(output_subdir, json_name)
                    with open(json_path, 'w') as json_file:
                        json.dump(meta, json_file, indent=4)
                    print(f"Exported metadata for {filename} to {json_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# After processing all files, create and save the combined fluorescence CSV
if fluorescence_data:
    combined_df = pd.concat(fluorescence_data, axis=1)
    combined_path = os.path.join(output_dir, 'fluorescence.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined fluorescence data to {combined_path}")

# Sort plot_data_list by filename for consistent order
plot_data_list.sort(key=lambda x: x[0])

# Create stacked figure if there are plots
if plot_data_list:
    num_plots = len(plot_data_list)
    # Figure size in inches (cm to inches conversion)
    fig_width_cm = 5.5
    subplot_height_cm = 1.5
    fig_height_cm = subplot_height_cm * num_plots
    fig_width_in = fig_width_cm / 2.54
    fig_height_in = fig_height_cm / 2.54

    fig, axs = plt.subplots(num_plots, 1, figsize=(fig_width_in, fig_height_in), sharex=True, gridspec_kw={'hspace': 0})

    if num_plots == 1:
        axs = [axs]  # Make it iterable

    for i, (filename, fitc_data, pb_data) in enumerate(plot_data_list):
        ax = axs[i]

        # Compute bandwidth for FITC
        std_fitc = np.std(fitc_data)
        if std_fitc == 0:
            std_fitc = 1
        bw_fitc = 10 / std_fitc

        # Compute bandwidth for Pacific Blue
        std_pb = np.std(pb_data)
        if std_pb == 0:
            std_pb = 1
        bw_pb = 10 / std_pb

        # Set up KDEs
        kde_fitc = gaussian_kde(fitc_data, bw_method=bw_fitc)
        kde_pb = gaussian_kde(pb_data, bw_method=bw_pb)

        # Generate points (clip to 0-800)
        x = np.linspace(0, 800, 500)

        y_fitc = kde_fitc(x) * 10000
        y_pb = kde_pb(x) * 10000

        # Add semitransparent shadows (fills) under the lines
        ax.fill_between(x, 0, y_fitc, color='green', alpha=0.3)
        ax.fill_between(x, 0, y_pb, color='blue', alpha=0.3)

        # Plot the lines on top with half the default linewidth (default is ~1.5, so 0.75)
        ax.plot(x, y_fitc, color='green', linewidth=0.75)
        ax.plot(x, y_pb, color='blue', linewidth=0.75)

        ax.set_xlim(0, 800)
        ax.set_ylim(0, 400)

        # No y-axis ticks or labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')

        # No grid, no legend, no title

        # Hide x-axis for all but the last
        if i < num_plots - 1:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            # Explicitly set x-ticks and labels for the last plot
            ax.set_xticks(np.arange(0, 801, 200))
            ax.set_xticklabels(['0', '200', '400', '600', '800'])

    # Save the combined figure automatically
    combined_plot_path = os.path.join(output_dir, 'combined_density.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined stacked density plot to {combined_plot_path}")

    # Optionally display interactively (comment out if not needed)
    # plt.show()

print("Batch conversion and analysis complete.")