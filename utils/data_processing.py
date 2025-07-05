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



def load_and_segment_data(dataset_path, 
                                           min_fragment_size=375,
                                           target_fragment_size=1250,
                                           val_ratio=0.2, 
                                           test_ratio=0.2, 
                                           random_seed=42):
    """
    Load data and create proper train/validation/test splits ensuring no patient leakage
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    min_fragment_size : int
        Minimum acceptable fragment size
    target_fragment_size : int
        Target fragment size
    val_ratio : float
        Ratio of patients for validation set (0.2 = 20%)
    test_ratio : float  
        Ratio of patients for test set (0.2 = 20%)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (train_data, val_data, test_data) where each is 
        (segments, labels, patient_ids, filenames)
    """
    import random
    from sklearn.model_selection import train_test_split
    
    random.seed(random_seed)
    
    # Step 1: Load all data grouped by patients
    all_patient_data = {}  # patient_id -> {'segments': [], 'labels': [], 'filenames': [], 'condition': 'AF'/'Healthy'}
    
    # Process AF patients
    af_path = os.path.join(dataset_path, "AF_Patients")
    if os.path.exists(af_path):
        af_files = [f for f in os.listdir(af_path) if f.endswith(".csv")]
        
        for filename in af_files:
            patient_id = extract_patient_id(filename)
            file_path = os.path.join(af_path, filename)
            
            try:
                data = pd.read_csv(file_path)
                ppg_data = data['PPG'].values
                segments = create_segments_from_signal(ppg_data, min_fragment_size, target_fragment_size)
                
                if patient_id not in all_patient_data:
                    all_patient_data[patient_id] = {
                        'segments': [], 
                        'labels': [], 
                        'filenames': [], 
                        'condition': 'AF'
                    }
                
                all_patient_data[patient_id]['segments'].extend(segments)
                all_patient_data[patient_id]['labels'].extend([1] * len(segments))  # AF = 1
                all_patient_data[patient_id]['filenames'].extend([filename] * len(segments))
                
                print(f"Processed AF file: {filename}, Patient: {patient_id}, Segments: {len(segments)}")
                
            except Exception as e:
                print(f"Error processing AF file {filename}: {e}")
    
    # Process Healthy patients
    healthy_path = os.path.join(dataset_path, "Healthy_Patients")
    if os.path.exists(healthy_path):
        healthy_files = [f for f in os.listdir(healthy_path) if f.endswith(".csv")]
        
        for filename in healthy_files:
            patient_id = extract_patient_id(filename)
            file_path = os.path.join(healthy_path, filename)
            
            try:
                data = pd.read_csv(file_path)
                ppg_data = data['PPG'].values
                segments = create_segments_from_signal(ppg_data, min_fragment_size, target_fragment_size)
                
                if patient_id not in all_patient_data:
                    all_patient_data[patient_id] = {
                        'segments': [], 
                        'labels': [], 
                        'filenames': [], 
                        'condition': 'Healthy'
                    }
                
                all_patient_data[patient_id]['segments'].extend(segments)
                all_patient_data[patient_id]['labels'].extend([0] * len(segments))  # Healthy = 0
                all_patient_data[patient_id]['filenames'].extend([filename] * len(segments))
                
                print(f"Processed Healthy file: {filename}, Patient: {patient_id}, Segments: {len(segments)}")
                
            except Exception as e:
                print(f"Error processing Healthy file {filename}: {e}")
    
    # Step 2: Determine patient-level labels for stratification
    patients = list(all_patient_data.keys())
    patient_labels = []
    
    for patient_id in patients:
        # Patient label based on condition (more robust than majority vote)
        condition = all_patient_data[patient_id]['condition']
        patient_label = 1 if condition == 'AF' else 0
        patient_labels.append(patient_label)
    
    print(f"\nTotal unique patients: {len(patients)}")
    af_patients = sum(patient_labels)
    healthy_patients = len(patients) - af_patients
    print(f"AF patients: {af_patients}, Healthy patients: {healthy_patients}")
    
    # Step 3: Create stratified patient splits (train/val/test)
    # First split: separate test set
    train_val_patients, test_patients, train_val_labels, test_labels = train_test_split(
        patients, patient_labels, 
        test_size=test_ratio, 
        random_state=random_seed, 
        stratify=patient_labels
    )
    
    # Second split: separate validation from training
    adjusted_val_ratio = val_ratio / (1 - test_ratio)  # Adjust since we removed test set
    train_patients, val_patients, _, _ = train_test_split(
        train_val_patients, train_val_labels,
        test_size=adjusted_val_ratio,
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    print(f"\nPatient split summary:")
    print(f"  Training: {len(train_patients)} patients ({len(train_patients)/len(patients)*100:.1f}%)")
    print(f"  Validation: {len(val_patients)} patients ({len(val_patients)/len(patients)*100:.1f}%)")
    print(f"  Test: {len(test_patients)} patients ({len(test_patients)/len(patients)*100:.1f}%)")
    
    # Step 4: Create segment-level data for each split
    def create_split_data(patient_list, split_name):
        segments, labels, patient_ids, filenames = [], [], [], []
        af_count = 0
        
        for patient_id in patient_list:
            data = all_patient_data[patient_id]
            segments.extend(data['segments'])
            labels.extend(data['labels'])
            patient_ids.extend([patient_id] * len(data['segments']))
            filenames.extend(data['filenames'])
            
            if data['condition'] == 'AF':
                af_count += 1
        
        healthy_count = len(patient_list) - af_count
        af_segments = sum(labels)
        healthy_segments = len(labels) - af_segments
        
        print(f"{split_name} set:")
        print(f"  Patients: {af_count} AF, {healthy_count} Healthy")
        print(f"  Segments: {af_segments} AF ({af_segments/len(labels)*100:.1f}%), {healthy_segments} Healthy")
        
        return segments, labels, patient_ids, filenames
    
    train_data = create_split_data(train_patients, "Training")
    val_data = create_split_data(val_patients, "Validation")
    test_data = create_split_data(test_patients, "Test")
    
    return train_data, val_data, test_data


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
    Extract unique patient ID that distinguishes between AF and non-AF patients
    """
    import re
    
    try:
        if "non_af" in filename:
            # Healthy patient
            match = re.search(r'non_af_(\d+)', filename)
            if match:
                number = match.group(1)
                return f"HEALTHY_{number}"
        else:
            # AF patient 
            match = re.search(r'(?<!non_)af_(\d+)', filename)
            if match:
                number = match.group(1)
                return f"AF_{number}"
        
        return filename  # fallback
        
    except Exception as e:
        print(f"Error extracting patient ID from {filename}: {e}")
        return filename


def create_train_splits_with_cv(segments, labels, patient_ids, n_folds=5, random_state=42):
    """
    Create train splits for 5-fold cross-validation based on patient IDs
    (Only for training set - validation and test sets are held out)
    
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
    
    # Convert to numpy arrays
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    # Create dummy array for GroupKFold
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
        
        print(f"CV Fold {fold_counter}/{n_folds}:")
        print(f"  Train: {len(train_idx)} segments from {len(train_patients)} patients")
        print(f"  Val: {len(val_idx)} segments from {len(val_patients)} patients")
        
        # Check for patient leakage
        overlap = set(train_patients) & set(val_patients)
        if overlap:
            print(f"  WARNING: Patient overlap detected: {overlap}")
        
        # Check class distribution
        train_af = np.sum(labels[train_idx] == 1)
        val_af = np.sum(labels[val_idx] == 1)
        
        print(f"  Train AF: {train_af}/{len(train_idx)} ({train_af/len(train_idx)*100:.1f}%)")
        print(f"  Val AF: {val_af}/{len(val_idx)} ({val_af/len(val_idx)*100:.1f}%)")
        
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