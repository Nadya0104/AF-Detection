import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy import signal

# Define paths
base_dir = "Dataset"
af_dir = os.path.join(base_dir, "AF_Patients")
output_dir = "ECG_Plots"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of all CSV files in AF_Patients directory
csv_files = [f for f in os.listdir(af_dir) if f.endswith('.csv')]

# Process each CSV file
for i, csv_file in enumerate(csv_files, 1):
    if i > 19:  # Only process first 19 files
        break
        
    print(f"Processing file {i} of 19: {csv_file}")
    
    # Read the CSV file
    file_path = os.path.join(af_dir, csv_file)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue
    
    # Check column names (case-insensitive)
    required_columns = ['time', 'ecg']
    df_columns_lower = [col.lower() for col in df.columns]
    
    time_col = None
    ecg_col = None
    
    for req_col in required_columns:
        if req_col not in df_columns_lower:
            print(f"Warning: {csv_file} is missing the required column '{req_col}'")
        else:
            idx = df_columns_lower.index(req_col)
            if req_col == 'time':
                time_col = df.columns[idx]
            elif req_col == 'ecg':
                ecg_col = df.columns[idx]
    
    if time_col is None or ecg_col is None:
        print(f"Skipping {csv_file} due to missing required columns")
        continue
    
    # Create PDF file for this recording
    output_file = os.path.join(output_dir, f"ECG_Plot_{i}.pdf")
    
    # Calculate sampling rate
    times = df[time_col].values
    if len(times) > 1:
        sampling_rate = 1 / (times[1] - times[0])
    else:
        sampling_rate = 125  # Default assumption
    
    # Calculate samples per 30 seconds
    samples_per_30sec = int(30 * sampling_rate)
    
    # Create a PDF with 40 subplots (one for each 30-second segment)
    with PdfPages(output_file) as pdf:
        # Process each 30-second segment
        for segment in range(40):
            # Calculate start and end indices for this segment
            start_idx = segment * samples_per_30sec
            end_idx = min((segment + 1) * samples_per_30sec, len(df))
            
            if start_idx >= len(df):
                print(f"Warning: Recording {csv_file} is shorter than {segment/2 + 0.5} minutes")
                break
            
            # Extract ECG data for this segment
            ecg_data = df[ecg_col].values[start_idx:end_idx]
            time_data = df[time_col].values[start_idx:end_idx]
            
            # Apply signal processing to make ECG look more standard
            if len(ecg_data) > 0:
                # 1. Apply bandpass filter (0.5-40Hz) - typical for ECG
                nyquist = sampling_rate / 2
                low_cutoff = 0.5 / nyquist
                high_cutoff = 40.0 / nyquist
                b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='band')
                try:
                    ecg_filtered = signal.filtfilt(b, a, ecg_data)
                except Exception as e:
                    print(f"Warning: Could not filter data: {e}")
                    ecg_filtered = ecg_data
                
                # 2. Normalize amplitude to typical ECG range (optional)
                # Standard ECG is typically calibrated at 10mm/mV
                # Here we just normalize to a reasonable visual range
                ecg_range = np.max(ecg_filtered) - np.min(ecg_filtered)
                if ecg_range > 0:
                    ecg_normalized = (ecg_filtered - np.min(ecg_filtered)) / ecg_range
                    ecg_normalized = ecg_normalized * 2.0 - 1.0  # Scale to [-1, 1]
                else:
                    ecg_normalized = ecg_filtered
                
                # Use processed signal
                plot_data = ecg_normalized
            else:
                plot_data = ecg_data
            
            # Create figure for this segment with increased width
            fig, ax = plt.figure(figsize=(18, 4)), plt.subplot(111)
            ax.plot(time_data, plot_data, 'b-', linewidth=0.8)
            
            # Set title and labels
            minute = segment // 2
            second_half = "30-60s" if segment % 2 else "0-30s"
            ax.set_title(f"ECG Signal - Minute {minute+1} ({second_half})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ECG Amplitude")
            
            # Set up ECG-like grid
            # Major grid - darker lines at 0.5mV intervals (typically 5mm paper squares)
            ax.grid(True, which='major', color='pink', linestyle='-', alpha=0.6)
            # Minor grid - lighter lines at 0.1mV intervals (typically 1mm paper squares)
            ax.grid(True, which='minor', color='pink', linestyle='-', alpha=0.2)
            ax.minorticks_on()
            
            # Set background color to mimic ECG paper
            ax.set_facecolor('#FFF9F9')  # Very light pink/beige
            
            # Tight layout
            plt.tight_layout()
            
            # Add to PDF
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"Created ECG plot file: {output_file}")

print("Processing complete. All ECG plots have been generated.")