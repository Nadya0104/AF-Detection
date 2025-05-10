"""
Data processing utilities for AF Detection
Streamlined for inference pipeline only
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import resample


# Constants
SAMPLING_RATE = 125  # Original sampling rate
TARGET_FRAGMENT_SIZE = 1250  # 10 seconds at 125 Hz
MIN_FRAGMENT_SIZE = 375  # 3 seconds at 125 Hz


def resample_signal(signal, target_rate, original_rate=SAMPLING_RATE):
    """
    Resample signal from original_rate to target_rate
    
    Parameters:
    -----------
    signal : numpy array
        Input signal
    target_rate : int
        Target sampling rate
    original_rate : int
        Original sampling rate (default 125 Hz)
    
    Returns:
    --------
    numpy array
        Resampled signal
    """
    if target_rate == original_rate:
        return signal
    
    num_samples = int(len(signal) * target_rate / original_rate)
    return resample(signal, num_samples)


def normalize_signal(signal):
    """
    Z-score normalization of the signal
    
    Parameters:
    -----------
    signal : numpy array
        Input signal
    
    Returns:
    --------
    numpy array
        Normalized signal
    """
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std > 0:
        return (signal - signal_mean) / signal_std
    return signal - signal_mean


def create_sliding_windows(signal, window_size, stride=None):
    """
    Create sliding windows from signal
    
    Parameters:
    -----------
    signal : numpy array
        Input signal
    window_size : int
        Size of each window
    stride : int, optional
        Stride between windows (default: window_size // 2)
    
    Returns:
    --------
    list
        List of windows
    """
    if stride is None:
        stride = window_size // 2
    
    windows = []
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i:i + window_size]
        if not np.isnan(window).any():
            windows.append(window)
    
    # Handle case where signal is too short
    if len(windows) == 0 and len(signal) > 0:
        if len(signal) < window_size:
            # Pad signal
            padded = np.zeros(window_size)
            padded[:len(signal)] = signal
            windows.append(padded)
        else:
            windows.append(signal[:window_size])
    
    return windows


def create_segments_from_signal(ppg_data, min_fragment_size, target_fragment_size):
    """Create segments from a single signal (for spectral analysis)"""
    segments = []
    current_segment = []
    
    for value in ppg_data:
        if np.isnan(value):
            # If we have a segment of sufficient length, add it
            if len(current_segment) >= min_fragment_size:
                segments.append(np.array(current_segment))
            current_segment = []
        else:
            current_segment.append(value)
            
            # If we've reached the target size, add segment
            if len(current_segment) == target_fragment_size:
                segments.append(np.array(current_segment))
                current_segment = []
    
    # Add the last segment if it's long enough
    if len(current_segment) >= min_fragment_size:
        segments.append(np.array(current_segment))
    
    return segments


def load_and_segment_data(dataset_path, min_fragment_size=MIN_FRAGMENT_SIZE, 
                         target_fragment_size=TARGET_FRAGMENT_SIZE):
    """
    Load all CSV files from the dataset and segment them for spectral analysis
    with patient ID tracking for proper cross-validation
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    min_fragment_size : int
        Minimum acceptable fragment size
    target_fragment_size : int
        Target fragment size
    
    Returns:
    --------
    tuple
        (segments, labels, patient_ids)
    """
    segments = []
    labels = []
    patient_ids = []
    
    # Process AF patients
    af_path = os.path.join(dataset_path, "AF_Patients")
    if os.path.exists(af_path):
        af_segments, af_labels, af_patient_ids = process_folder(
            af_path, 1, min_fragment_size, target_fragment_size
        )
        segments.extend(af_segments)
        labels.extend(af_labels)
        patient_ids.extend(af_patient_ids)
    
    # Process Healthy patients
    healthy_path = os.path.join(dataset_path, "Healthy_Patients")
    if os.path.exists(healthy_path):
        healthy_segments, healthy_labels, healthy_patient_ids = process_folder(
            healthy_path, 0, min_fragment_size, target_fragment_size
        )
        segments.extend(healthy_segments)
        labels.extend(healthy_labels)
        patient_ids.extend(healthy_patient_ids)
    
    return segments, labels, patient_ids


def process_folder(folder_path, label, min_fragment_size, target_fragment_size):
    """Process all files in a folder for spectral analysis with patient ID tracking"""
    segments = []
    labels = []
    patient_ids = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extract better patient ID from filename
            # For format "mimic_perform_af_001_data.csv" use the number as patient ID
            parts = filename.split('_')
            
            try:
                if "non_af" in filename:
                    # For non-AF files like "mimic_perform_non_af_001_data.csv"
                    parts = filename.split('non_af_')
                    if len(parts) > 1:
                        # Extract the number part (001)
                        patient_id = parts[1].split('_')[0]
                    else:
                        patient_id = filename
                elif "_af_" in filename:
                    # For AF files like "mimic_perform_af_001_data.csv"
                    parts = filename.split('_af_')
                    if len(parts) > 1:
                        # Extract the number part (001)
                        patient_id = parts[1].split('_')[0]
                    else:
                        patient_id = filename
                else:
                    # Fallback
                    patient_id = filename
            except:
                # Fallback for any extraction errors
                patient_id = filename
                
            file_path = os.path.join(folder_path, filename)
            try:
                data = pd.read_csv(file_path)
                ppg_data = data['PPG'].values
                
                # Create segments for spectral analysis
                file_segments = create_segments_from_signal(
                    ppg_data, min_fragment_size, target_fragment_size
                )
                
                # Add segments and tracking info
                segments.extend(file_segments)
                labels.extend([label] * len(file_segments))
                patient_ids.extend([patient_id] * len(file_segments))
                
                print(f"Processed file: {filename}, Patient ID: {patient_id}, Segments: {len(file_segments)}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    return segments, labels, patient_ids


class PPGInferenceDataset(Dataset):
    """Dataset for PPG signals for transformer inference"""
    
    def __init__(self, ppg_signals, context_length=500, sample_rate=50):
        self.data = []
        
        for signal in ppg_signals:
            # Resample if needed
            if sample_rate != SAMPLING_RATE:
                signal = resample_signal(signal, sample_rate)
            
            # Normalize
            signal = normalize_signal(signal)
            
            # Create windows
            windows = create_sliding_windows(signal, context_length, stride=250)
            self.data.extend(windows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)