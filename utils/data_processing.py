"""
Data processing utilities for AF Detection
Common functions for both transformer and spectral analysis
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


def augment_signal(windows, window_size, num_augmentations=3):
    """
    Data augmentation for PPG signals
    
    Parameters:
    -----------
    windows : list
        List of signal windows
    window_size : int
        Size of each window
    num_augmentations : int
        Number of augmentations per window
    
    Returns:
    --------
    list
        Augmented windows
    """
    aug_windows = []
    
    for window in windows[:len(windows)//10]:  # Only augment a subset
        # Time shifting
        shift = np.random.randint(1, window_size // 10)
        aug_windows.append(np.roll(window, shift))
        
        # Add small Gaussian noise
        noise_level = 0.05
        aug_windows.append(window + np.random.normal(0, noise_level, window.shape))
        
        # Small amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        aug_windows.append(window * scale)
    
    return aug_windows


def load_and_segment_data(dataset_path, min_fragment_size=MIN_FRAGMENT_SIZE, 
                         target_fragment_size=TARGET_FRAGMENT_SIZE,
                         for_transformer=False, transformer_window_size=500,
                         transformer_stride=50):
    """
    Load all CSV files from the dataset and segment them
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    min_fragment_size : int
        Minimum acceptable fragment size (for spectral analysis)
    target_fragment_size : int
        Target fragment size (for spectral analysis)
    for_transformer : bool
        If True, create windows for transformer model
    transformer_window_size : int
        Window size for transformer model
    transformer_stride : int
        Stride for transformer model windows
    
    Returns:
    --------
    tuple
        (segments, labels) or ((segments, windows), labels) if for_transformer
    """
    segments = []  # For spectral analysis
    windows = []   # For transformer model
    labels = []
    
    # Process AF patients
    af_path = os.path.join(dataset_path, "AF_Patients")
    if os.path.exists(af_path):
        af_data = process_folder(af_path, 1, min_fragment_size, target_fragment_size,
                               for_transformer, transformer_window_size, transformer_stride)
        if for_transformer:
            af_segments, af_windows, af_labels = af_data
            segments.extend(af_segments)
            windows.extend(af_windows)
        else:
            af_segments, af_labels = af_data
            segments.extend(af_segments)
        labels.extend(af_labels)
    
    # Process Healthy patients
    healthy_path = os.path.join(dataset_path, "Healthy_Patients")
    if os.path.exists(healthy_path):
        healthy_data = process_folder(healthy_path, 0, min_fragment_size, target_fragment_size,
                                    for_transformer, transformer_window_size, transformer_stride)
        if for_transformer:
            healthy_segments, healthy_windows, healthy_labels = healthy_data
            segments.extend(healthy_segments)
            windows.extend(healthy_windows)
        else:
            healthy_segments, healthy_labels = healthy_data
            segments.extend(healthy_segments)
        labels.extend(healthy_labels)
    
    if for_transformer:
        # Return lists instead of numpy arrays with dtype=object
        return (segments, windows), labels
    else:
        return segments, labels


def process_folder(folder_path, label, min_fragment_size, target_fragment_size,
                  for_transformer=False, transformer_window_size=500, transformer_stride=50):
    """Process all files in a folder"""
    segments = []
    windows = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                data = pd.read_csv(file_path)
                ppg_data = data['PPG'].values
                
                # Create segments for spectral analysis
                file_segments = create_segments_from_signal(ppg_data, min_fragment_size, 
                                                          target_fragment_size)
                segments.extend(file_segments)
                
                # Create windows for transformer if needed
                if for_transformer:
                    # Clean signal (remove NaN)
                    clean_signal = ppg_data[~np.isnan(ppg_data)]
                    
                    if len(clean_signal) > 0:
                        # Resample to 50 Hz for transformer
                        resampled_signal = resample_signal(clean_signal, 50)
                        
                        # Normalize
                        normalized_signal = normalize_signal(resampled_signal)
                        
                        # Create windows
                        signal_windows = create_sliding_windows(normalized_signal, 
                                                              transformer_window_size,
                                                              transformer_stride)
                        windows.extend(signal_windows)
                        labels.extend([label] * len(signal_windows))
                else:
                    # For spectral analysis, labels match segments
                    labels.extend([label] * len(file_segments))    
                
                print(f"Processed file: {filename}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    if for_transformer:
        return segments, windows, labels
    else:
        return segments, labels


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


class BasePPGDataset(Dataset):
    """Base class for PPG datasets with common functionality"""
    
    def resample_signal(self, signal, target_rate):
        """Wrapper for resample_signal function"""
        return resample_signal(signal, target_rate)
    
    def normalize_signal(self, signal):
        """Wrapper for normalize_signal function"""
        return normalize_signal(signal)
    
    def create_sliding_windows(self, signal, window_size, stride=None):
        """Wrapper for create_sliding_windows function"""
        return create_sliding_windows(signal, window_size, stride)


class PPGInferenceDataset(BasePPGDataset):
    """Dataset for PPG signals for transformer inference"""
    
    def __init__(self, ppg_signals, context_length=500, sample_rate=50):
        self.data = []
        
        for signal in ppg_signals:
            # Resample if needed
            if sample_rate != SAMPLING_RATE:
                signal = self.resample_signal(signal, sample_rate)
            
            # Normalize
            signal = self.normalize_signal(signal)
            
            # Create windows
            windows = self.create_sliding_windows(signal, context_length, stride=250)
            self.data.extend(windows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


class PPGTrainingDataset(BasePPGDataset):
    """Dataset for training transformer model"""
    
    def __init__(self, folder_path, context_length=500, sample_rate=50, augment=False):
        self.data = []
        self.labels = []
        
        # Load data using the unified function
        (_, windows), labels = load_and_segment_data(
            folder_path, 
            for_transformer=True,
            transformer_window_size=context_length,
            transformer_stride=50
        )
        
        # Convert to appropriate format - handle windows properly
        valid_windows = []
        valid_labels = []
        
        # Check each window and ensure it has the correct shape
        for i, window in enumerate(windows):
            # Ensure window is a numpy array
            window = np.array(window, dtype=np.float32)

            if len(window) == context_length:
                valid_windows.append(window)
                valid_labels.append(labels[i])
            else:
                # Handle windows with incorrect length by padding or truncating
                if len(window) < context_length:
                    # Pad with zeros
                    padded = np.zeros(context_length)
                    padded[:len(window)] = window
                    valid_windows.append(padded)
                else:
                    # Truncate
                    valid_windows.append(window[:context_length])
                valid_labels.append(labels[i])
        
        self.data = valid_windows
        self.labels = valid_labels
        
        # Apply augmentation if requested
        if augment:
            augmented_data = []
            augmented_labels = []
            
            # Group by label for balanced augmentation
            af_indices = [i for i, label in enumerate(self.labels) if label == 1]
            normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
            
            # Augment AF data
            if len(af_indices) > 0:
                af_windows = [self.data[i] for i in af_indices]
                af_aug = augment_signal(af_windows, context_length)
                # Ensure augmented data is also properly formatted
                af_aug = [np.array(w, dtype=np.float32) for w in af_aug]
                augmented_data.extend(af_aug)
                augmented_labels.extend([1] * len(af_aug))
            
            # Augment Normal data
            if len(normal_indices) > 0:
                normal_windows = [self.data[i] for i in normal_indices]
                normal_aug = augment_signal(normal_windows, context_length)
                # Ensure augmented data is also properly formatted
                normal_aug = [np.array(w, dtype=np.float32) for w in normal_aug]
                augmented_data.extend(normal_aug)
                augmented_labels.extend([0] * len(normal_aug))
            
            # Add augmented data
            self.data.extend(augmented_data)
            self.labels.extend(augmented_labels)
        
        # Convert to tensors - now all windows should have the same shape
        if len(self.data) > 0:
            # Create a proper numpy array first
            data_array = np.stack(self.data, axis=0)
            self.data = torch.tensor(data_array, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
        else:
            self.data = torch.empty((0, context_length), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]