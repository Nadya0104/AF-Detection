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
    """
    Create segments from a single signal (for spectral analysis)
    
    Parameters:
    -----------
    ppg_data : numpy array
        Input PPG signal
    min_fragment_size : int
        Minimum acceptable fragment size
    target_fragment_size : int
        Target fragment size
    
    Returns:
    --------
    list
        List of segments
    """
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
                         target_fragment_size=TARGET_FRAGMENT_SIZE, test_ratio=0.2, random_seed=42):
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
    test_ratio : float
        Ratio of patients to reserve for testing (e.g., 0.2 = 20%)
    random_seed : int
        Random seed for reproducible splitting
    
    Returns:
    --------
    tuple
        (train_segments, train_labels, train_patient_ids, train_filenames,
         test_segments, test_labels, test_patient_ids, test_filenames)
    """
    import random
    random.seed(random_seed)
    
    # Initialize lists for data collection
    all_segments = []
    all_labels = []
    all_patient_ids = []
    all_filenames = []
    
    # Process AF patients
    af_path = os.path.join(dataset_path, "AF_Patients")
    if os.path.exists(af_path):
        # Get all CSV files in the folder
        af_files = [f for f in os.listdir(af_path) if f.endswith(".csv")]
        
        # Process each file and track patient IDs
        for filename in af_files:
            patient_id = extract_patient_id(filename)
            
            file_path = os.path.join(af_path, filename)
            try:
                data = pd.read_csv(file_path)
                ppg_data = data['PPG'].values
                
                # Create segments
                file_segments = create_segments_from_signal(
                    ppg_data, min_fragment_size, target_fragment_size
                )
                
                # Add segments and tracking info
                all_segments.extend(file_segments)
                all_labels.extend([1] * len(file_segments))  # 1 for AF
                all_patient_ids.extend([patient_id] * len(file_segments))
                all_filenames.extend([filename] * len(file_segments))
                
                print(f"Processed AF file: {filename}, Patient ID: {patient_id}, Segments: {len(file_segments)}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    # Process Healthy patients
    healthy_path = os.path.join(dataset_path, "Healthy_Patients")
    if os.path.exists(healthy_path):
        # Get all CSV files in the folder
        healthy_files = [f for f in os.listdir(healthy_path) if f.endswith(".csv")]
        
        # Process each file and track patient IDs
        for filename in healthy_files:
            patient_id = extract_patient_id(filename)
            
            file_path = os.path.join(healthy_path, filename)
            try:
                data = pd.read_csv(file_path)
                ppg_data = data['PPG'].values
                
                # Create segments
                file_segments = create_segments_from_signal(
                    ppg_data, min_fragment_size, target_fragment_size
                )
                
                # Add segments and tracking info
                all_segments.extend(file_segments)
                all_labels.extend([0] * len(file_segments))  # 0 for healthy
                all_patient_ids.extend([patient_id] * len(file_segments))
                all_filenames.extend([filename] * len(file_segments))
                
                print(f"Processed healthy file: {filename}, Patient ID: {patient_id}, Segments: {len(file_segments)}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    # Create patient-based train/test split
    unique_patients = list(set(all_patient_ids))
    print(f"Total unique patients: {len(unique_patients)}")
    
    # Randomly select test patients
    num_test_patients = max(1, int(len(unique_patients) * test_ratio))
    test_patients = random.sample(unique_patients, num_test_patients)
    print(f"Selected {len(test_patients)} patients for testing: {test_patients}")
    
    # Split data based on patient IDs
    train_segments = []
    train_labels = []
    train_patient_ids = []
    train_filenames = []
    
    test_segments = []
    test_labels = []
    test_patient_ids = []
    test_filenames = []
    
    for i, patient_id in enumerate(all_patient_ids):
        if patient_id in test_patients:
            test_segments.append(all_segments[i])
            test_labels.append(all_labels[i])
            test_patient_ids.append(patient_id)
            test_filenames.append(all_filenames[i])
        else:
            train_segments.append(all_segments[i])
            train_labels.append(all_labels[i])
            train_patient_ids.append(patient_id)
            train_filenames.append(all_filenames[i])
    
    # Print split information
    print(f"Train set: {len(train_segments)} segments from {len(set(train_patient_ids))} patients")
    print(f"Test set: {len(test_segments)} segments from {len(set(test_patient_ids))} patients")
    
    # Check class distribution
    train_af = sum(train_labels)
    train_healthy = len(train_labels) - train_af
    test_af = sum(test_labels)
    test_healthy = len(test_labels) - test_af
    
    print(f"Train set: {train_af} AF ({train_af/len(train_labels)*100:.1f}%), {train_healthy} healthy")
    print(f"Test set: {test_af} AF ({test_af/len(test_labels)*100:.1f}%), {test_healthy} healthy")
    
    return (train_segments, train_labels, train_patient_ids, train_filenames,
            test_segments, test_labels, test_patient_ids, test_filenames)


def process_folder(folder_path, label, min_fragment_size, target_fragment_size, exclude_last=True):
    """
    Process all files in a folder for spectral analysis with patient ID tracking
    Optionally excludes the last file for testing purposes
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing CSV files
    label : int
        Label for this folder (0 for normal, 1 for AF)
    min_fragment_size : int
        Minimum acceptable fragment size
    target_fragment_size : int
        Target fragment size
    exclude_last : bool
        Whether to exclude the last file (for test set)
    
    Returns:
    --------
    tuple
        (segments, labels, patient_ids, filenames)
    """
    segments = []
    labels = []
    patient_ids = []
    filenames = []
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    # Sort files to ensure consistent "last file" for exclusion
    csv_files.sort()
    
    # If exclude_last is True, remove the last file
    if exclude_last and len(csv_files) > 1:
        test_file = csv_files[-1]
        csv_files = csv_files[:-1]
        # print(f"Excluding file for testing: {test_file}")
    
    # Process each file
    for filename in csv_files:
        # Extract patient ID from filename
        patient_id = extract_patient_id(filename)
        
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
            filenames.extend([filename] * len(file_segments))
            
            print(f"Processed file: {filename}, Patient ID: {patient_id}, Segments: {len(file_segments)}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return segments, labels, patient_ids, filenames

def extract_patient_id(filename):
    """
    Extract patient ID from filename
    
    Parameters:
    -----------
    filename : str
        Filename to extract patient ID from
    
    Returns:
    --------
    str
        Patient ID
    """
    try:
        if "non_af" in filename:
            # For non-AF files like "mimic_perform_non_af_001_data.csv"
            parts = filename.split('non_af_')
            if len(parts) > 1:
                # Extract the number part (001)
                patient_id = parts[1].split('_')[0]
                return patient_id
            else:
                return filename
        elif "_af_" in filename:
            # For AF files like "mimic_perform_af_001_data.csv"
            parts = filename.split('_af_')
            if len(parts) > 1:
                # Extract the number part (001)
                patient_id = parts[1].split('_')[0]
                return patient_id
            else:
                return filename
        else:
            # Fallback
            return filename
    except Exception as e:
        # Log the exception for debugging
        print(f"Error extracting patient ID from {filename}: {e}")
        # Fallback for any extraction errors
        return filename


def create_train_val_splits_with_cv(segments, labels, patient_ids, n_folds=5, random_state=42):
    """
    Create train/validation splits for 5-fold cross-validation based on patient IDs
    
    Parameters:
    -----------
    segments : list
        List of PPG segments (can have different lengths)
    labels : list or numpy array
        List of labels (0 for normal, 1 for AF)
    patient_ids : list or numpy array
        List of patient IDs
    n_folds : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    list
        List of (train_idx, val_idx) tuples for each fold
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold
    
    # Convert to numpy arrays for labels and patient_ids
    # (but keep segments as a list since they can have different lengths)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    # Create dummy array for GroupKFold (one row per segment)
    # GroupKFold doesn't actually use X, just its shape
    X_dummy = np.zeros((len(segments), 1))
    
    # Create group k-fold
    group_kfold = GroupKFold(n_splits=n_folds)
    
    # Generate folds
    cv_folds = []
    fold_counter = 1
    
    for train_idx, val_idx in group_kfold.split(X_dummy, labels, patient_ids):
        cv_folds.append((train_idx, val_idx))
        
        # Debug information
        train_patients = np.unique(patient_ids[train_idx])
        val_patients = np.unique(patient_ids[val_idx])
        
        print(f"Fold {fold_counter}/{n_folds}: Train on {len(train_idx)} segments from {len(train_patients)} patients, "
              f"validate on {len(val_idx)} segments from {len(val_patients)} patients")
        
        # Check class distribution
        train_af = np.sum(labels[train_idx] == 1)
        val_af = np.sum(labels[val_idx] == 1)
        
        print(f"  Train: {train_af} AF segments ({train_af/len(train_idx)*100:.1f}%), "
              f"Val: {val_af} AF segments ({val_af/len(val_idx)*100:.1f}%)")
        
        fold_counter += 1
    
    return cv_folds


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