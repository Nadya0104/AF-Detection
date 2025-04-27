import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import math

# Create output directory if it doesn't exist
output_dir = "ECG_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dataset path and recording info
dataset_path = r"Dataset\AF_WFDB"
recording_prefix = "mimic_perform_af_"
num_recordings = 19

# Parameters for plotting
seconds_per_segment = 20
segments_per_minute = 3  # 3 segments of 20 seconds = 1 minute
plots_per_page = 9  # 3 segments × 3 minutes = 9 plots per page
total_minutes = 20  # Total duration of each recording in minutes

for recording_num in range(1, num_recordings + 1):
    # Format recording number with leading zeros
    recording_id = f"{recording_num:03d}"
    recording_name = f"{recording_prefix}{recording_id}"
    recording_path = os.path.join(dataset_path, recording_name)
    output_filename = os.path.join(output_dir, f"ECG_{recording_name}.pdf")
    
    print(f"Processing recording {recording_id}...")
    
    try:
        # Get record information
        record_info = wfdb.rdheader(recording_path)
        fs = record_info.fs  # Sampling frequency
        
        # Read the record
        record = wfdb.rdrecord(recording_path)
        
        # Check if we have at least 3 signals (assuming ECG is the second one)
        if record.n_sig < 2:
            print(f"Warning: Recording {recording_id} has fewer than 2 signals. Skipping.")
            continue
        
        # Extract ECG signal (second signal, index 1)
        ecg_signal = record.p_signal[:, 1]
        total_samples = len(ecg_signal)
        
        # Calculate number of samples per segment
        samples_per_segment = int(seconds_per_segment * fs)
        
        # Calculate number of pages needed
        total_segments = math.ceil(total_samples / samples_per_segment)
        total_pages = math.ceil(total_segments / plots_per_page)
        
        # Set height of each subplot for better visualization
        subplot_height = 2  # in inches
        
        # Create PDF
        with PdfPages(output_filename) as pdf:
            for page in range(total_pages):
                fig = plt.figure(figsize=(15, 20))  # Portrait orientation with wider plots
                fig.suptitle(f"Recording {recording_id} - ECG Signal", fontsize=16)
                
                # Process each plot on this page
                for plot_idx in range(plots_per_page):
                    segment_idx = page * plots_per_page + plot_idx
                    if segment_idx >= total_segments:
                        break
                    
                    # Calculate start and end samples for this segment
                    start_sample = segment_idx * samples_per_segment
                    end_sample = min((segment_idx + 1) * samples_per_segment, total_samples)
                    
                    # Calculate which minute this segment belongs to
                    current_minute = (segment_idx * seconds_per_segment) // 60 + 1
                    segment_in_minute = (segment_idx % segments_per_minute) + 1
                    
                    # Create subplot - one plot per row (9 rows, 1 column)
                    ax = fig.add_subplot(9, 1, plot_idx + 1)
                    
                    # Plot ECG segment
                    time_axis = np.arange(start_sample, end_sample) / fs
                    ax.plot(time_axis, ecg_signal[start_sample:end_sample])
                    
                    # Set title and labels
                    if segment_in_minute == 1:
                        ax.set_title(f"Minute {current_minute} - Segment {segment_in_minute}", fontsize=12)
                    else:
                        ax.set_title(f"Segment {segment_in_minute}", fontsize=12)
                    
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True)
                    
                    # Remove extra whitespace between subplots
                    plt.subplots_adjust(hspace=0.4)
                
                plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust to leave room for suptitle
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"Successfully created PDF for recording {recording_id}")
    
    except Exception as e:
        print(f"Error processing recording {recording_id}: {str(e)}")

print("Processing complete. PDFs saved in the ECG_plots folder.")