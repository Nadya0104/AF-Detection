"""
Complete Data Processing Module with Model-Specific Preprocessors
Two classes: SpectralPreprocessor and TransformerPreprocessor
Each handles both training and inference with identical logic
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import train_test_split, GroupKFold
import re


class SpectralPreprocessor:
    """Handles all preprocessing for spectral analysis model"""
    
    def __init__(self, min_fragment_size=375, target_fragment_size=1250):
        self.min_fragment_size = min_fragment_size  # 3 seconds at 125Hz
        self.target_fragment_size = target_fragment_size  # 10 seconds at 125Hz
    
    def load_training_data(self, dataset_path, val_ratio=0.2, test_ratio=0.2, random_seed=42):
        """Load and split training data with proper patient-based splitting"""
        
        # Load all data grouped by patients
        all_patient_data = {}
        
        # Process AF patients
        af_path = os.path.join(dataset_path, "AF_Patients")
        if os.path.exists(af_path):
            print("Processing AF patients...")
            for filename in os.listdir(af_path):
                if filename.endswith(".csv"):
                    self._process_file(af_path, filename, all_patient_data, label=1)
        
        # Process Healthy patients
        healthy_path = os.path.join(dataset_path, "Healthy_Patients")
        if os.path.exists(healthy_path):
            print("Processing Healthy patients...")
            for filename in os.listdir(healthy_path):
                if filename.endswith(".csv"):
                    self._process_file(healthy_path, filename, all_patient_data, label=0)
        
        # Split patients into train/val/test
        return self._split_data(all_patient_data, val_ratio, test_ratio, random_seed)
    
    def process_inference_data(self, ppg_signal):
        """Process new PPG signal for inference - SAME logic as training"""
        segments = self._create_segments(ppg_signal)
        print(f"Created {len(segments)} segments for inference")
        return segments
    
    def create_cv_folds(self, segments, labels, patient_ids, n_folds=5, random_state=42):
        """Create cross-validation folds based on patient IDs"""
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
            
            fold_counter += 1
        
        return cv_folds
    
    def _process_file(self, folder_path, filename, all_patient_data, label):
        """Process single CSV file"""
        patient_id = self._extract_patient_id(filename)
        file_path = os.path.join(folder_path, filename)
        
        try:
            data = pd.read_csv(file_path)
            ppg_data = data['PPG'].values
            
            # Create segments using NaN-aware segmentation
            segments = self._create_segments(ppg_data)
            
            if patient_id not in all_patient_data:
                all_patient_data[patient_id] = {
                    'segments': [], 'labels': [], 'patient_ids': [], 'filenames': []
                }
            
            all_patient_data[patient_id]['segments'].extend(segments)
            all_patient_data[patient_id]['labels'].extend([label] * len(segments))
            all_patient_data[patient_id]['patient_ids'].extend([patient_id] * len(segments))
            all_patient_data[patient_id]['filenames'].extend([filename] * len(segments))
            
            print(f"  {filename} -> {len(segments)} segments")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    def _create_segments(self, ppg_data):
        """Create segments from PPG signal - handles NaN properly"""
        segments = []
        current_segment = []
        
        for value in ppg_data:
            if np.isnan(value):
                # Hit NaN - save current segment if long enough
                if len(current_segment) >= self.min_fragment_size:
                    segments.append(np.array(current_segment))
                current_segment = []
            else:
                current_segment.append(value)
                # If reached target size, save and start new
                if len(current_segment) == self.target_fragment_size:
                    segments.append(np.array(current_segment))
                    current_segment = []
        
        # Handle last segment
        if len(current_segment) >= self.min_fragment_size:
            segments.append(np.array(current_segment))
        
        return segments
    
    def _split_data(self, all_patient_data, val_ratio, test_ratio, random_seed):
        """Split data by patients to prevent leakage"""
        patients = list(all_patient_data.keys())
        patient_labels = []
        
        # Determine patient-level labels
        for patient_id in patients:
            # Use first label (all segments from same patient have same label)
            patient_label = all_patient_data[patient_id]['labels'][0]
            patient_labels.append(patient_label)
        
        print(f"\nTotal patients: {len(patients)}")
        af_patients = sum(patient_labels)
        healthy_patients = len(patients) - af_patients
        print(f"AF patients: {af_patients}, Healthy patients: {healthy_patients}")
        
        # Split patients
        train_val_patients, test_patients = train_test_split(
            patients, test_size=test_ratio, random_state=random_seed, stratify=patient_labels
        )
        
        train_val_labels = [patient_labels[patients.index(p)] for p in train_val_patients]
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=adjusted_val_ratio, 
            random_state=random_seed, stratify=train_val_labels
        )
        
        print(f"Patient splits - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        
        # Convert to segment-level data
        train_data = self._create_split_data(train_patients, all_patient_data, "Training")
        val_data = self._create_split_data(val_patients, all_patient_data, "Validation")
        test_data = self._create_split_data(test_patients, all_patient_data, "Test")
        
        return train_data, val_data, test_data
    
    def _create_split_data(self, patient_list, all_patient_data, split_name):
        """Convert patient data to segment lists"""
        segments, labels, patient_ids, filenames = [], [], [], []
        
        for patient_id in patient_list:
            data = all_patient_data[patient_id]
            segments.extend(data['segments'])
            labels.extend(data['labels'])
            patient_ids.extend(data['patient_ids'])
            filenames.extend(data['filenames'])
        
        af_count = sum(labels)
        print(f"{split_name}: {len(segments)} segments ({af_count} AF, {len(segments)-af_count} Normal)")
        
        return segments, labels, patient_ids, filenames
    
    def _extract_patient_id(self, filename):
        """Extract unique patient ID"""
        if "non_af" in filename:
            match = re.search(r'non_af_(\d+)', filename)
            return f"HEALTHY_{match.group(1)}" if match else filename
        else:
            match = re.search(r'(?<!non_)af_(\d+)', filename)
            return f"AF_{match.group(1)}" if match else filename


class TransformerPreprocessor:
    """Handles all preprocessing for transformer model"""
    
    def __init__(self, context_length=500, stride=50, sample_rate=50):
        self.context_length = context_length  # Window size
        self.stride = stride  # Overlap between windows
        self.sample_rate = sample_rate  # Target sampling rate
    
    def load_training_data(self, dataset_path, val_ratio=0.2, test_ratio=0.2, random_seed=42):
        """Load and split training data with proper patient-based splitting"""
        
        # Load all data grouped by patients
        all_patient_data = {}
        
        # Process AF patients
        af_path = os.path.join(dataset_path, "AF_Patients")
        if os.path.exists(af_path):
            print("Processing AF patients...")
            for filename in os.listdir(af_path):
                if filename.endswith(".csv"):
                    self._process_file(af_path, filename, all_patient_data, label=1)
        
        # Process Healthy patients
        healthy_path = os.path.join(dataset_path, "Healthy_Patients")
        if os.path.exists(healthy_path):
            print("Processing Healthy patients...")
            for filename in os.listdir(healthy_path):
                if filename.endswith(".csv"):
                    self._process_file(healthy_path, filename, all_patient_data, label=0)
        
        # Split patients into train/val/test
        return self._split_data(all_patient_data, val_ratio, test_ratio, random_seed)
    
    def process_inference_data(self, ppg_signal):
        """Process new PPG signal for inference - SAME logic as training"""
        windows = self._create_windows(ppg_signal)
        print(f"Created {len(windows)} windows for inference")
        return windows
    
    def create_cv_folds(self, windows, labels, patient_ids, n_folds=5, random_state=42):
        """Create cross-validation folds based on patient IDs"""
        # Convert to numpy arrays
        labels = np.array(labels)
        patient_ids = np.array(patient_ids)
        
        # Create dummy array for GroupKFold
        X_dummy = np.zeros((len(windows), 1))
        
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
            print(f"  Train: {len(train_idx)} windows from {len(train_patients)} patients")
            print(f"  Val: {len(val_idx)} windows from {len(val_patients)} patients")
            
            # Check for patient leakage
            overlap = set(train_patients) & set(val_patients)
            if overlap:
                print(f"  WARNING: Patient overlap detected: {overlap}")
            
            fold_counter += 1
        
        return cv_folds
    
    def _process_file(self, folder_path, filename, all_patient_data, label):
        """Process single CSV file"""
        patient_id = self._extract_patient_id(filename)
        file_path = os.path.join(folder_path, filename)
        
        try:
            data = pd.read_csv(file_path)
            ppg_data = data['PPG'].values
            
            # Create windows using proper segmentation
            windows = self._create_windows(ppg_data)
            
            if patient_id not in all_patient_data:
                all_patient_data[patient_id] = {
                    'windows': [], 'labels': [], 'patient_ids': [], 'filenames': []
                }
            
            all_patient_data[patient_id]['windows'].extend(windows)
            all_patient_data[patient_id]['labels'].extend([label] * len(windows))
            all_patient_data[patient_id]['patient_ids'].extend([patient_id] * len(windows))
            all_patient_data[patient_id]['filenames'].extend([filename] * len(windows))
            
            print(f"  {filename} -> {len(windows)} windows")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    def _create_windows(self, ppg_data):
        """Create windows from PPG signal - proper NaN handling"""
        # Step 1: Create clean segments (no NaN gaps)
        segments = self._create_segments(ppg_data)
        
        # Step 2: Create windows from each clean segment
        all_windows = []
        for segment in segments:
            # Resample to target rate
            if self.sample_rate != 125:  # Original sampling rate
                segment = self._resample_signal(segment, self.sample_rate)
            
            # Normalize
            segment = self._normalize_signal(segment)
            
            # Create sliding windows from this segment
            segment_windows = self._sliding_windows(segment)
            all_windows.extend(segment_windows)
        
        return all_windows
    
    def _create_segments(self, ppg_data):
        """Create clean segments by splitting on NaN"""
        segments = []
        current_segment = []
        
        for value in ppg_data:
            if np.isnan(value):
                # Hit NaN - save current segment if long enough
                if len(current_segment) >= self.context_length:  # At least 1 window
                    segments.append(np.array(current_segment))
                current_segment = []
            else:
                current_segment.append(value)
                # If too long, split it
                if len(current_segment) == self.context_length * 4:  # Max 4 windows
                    segments.append(np.array(current_segment))
                    current_segment = []
        
        # Handle last segment
        if len(current_segment) >= self.context_length:
            segments.append(np.array(current_segment))
        
        return segments
    
    def _sliding_windows(self, signal):
        """Create sliding windows from clean signal"""
        windows = []
        for i in range(0, len(signal) - self.context_length + 1, self.stride):
            window = signal[i:i + self.context_length]
            windows.append(window)
        
        return windows
    
    def _resample_signal(self, signal, target_rate):
        """Resample signal to target rate"""
        original_rate = 125
        if target_rate == original_rate:
            return signal
        num_samples = int(len(signal) * target_rate / original_rate)
        return resample(signal, num_samples)
    
    def _normalize_signal(self, signal):
        """Z-score normalize signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            return (signal - mean) / std
        return signal - mean
    
    def _split_data(self, all_patient_data, val_ratio, test_ratio, random_seed):
        """Split data by patients to prevent leakage"""
        patients = list(all_patient_data.keys())
        patient_labels = []
        
        # Determine patient-level labels
        for patient_id in patients:
            # Use first label (all windows from same patient have same label)
            patient_label = all_patient_data[patient_id]['labels'][0]
            patient_labels.append(patient_label)
        
        print(f"\nTotal patients: {len(patients)}")
        af_patients = sum(patient_labels)
        healthy_patients = len(patients) - af_patients
        print(f"AF patients: {af_patients}, Healthy patients: {healthy_patients}")
        
        # Split patients
        train_val_patients, test_patients = train_test_split(
            patients, test_size=test_ratio, random_state=random_seed, stratify=patient_labels
        )
        
        train_val_labels = [patient_labels[patients.index(p)] for p in train_val_patients]
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=adjusted_val_ratio, 
            random_state=random_seed, stratify=train_val_labels
        )
        
        print(f"Patient splits - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        
        # Convert to window-level data
        train_data = self._create_split_data(train_patients, all_patient_data, "Training")
        val_data = self._create_split_data(val_patients, all_patient_data, "Validation")
        test_data = self._create_split_data(test_patients, all_patient_data, "Test")
        
        return train_data, val_data, test_data
    
    def _create_split_data(self, patient_list, all_patient_data, split_name):
        """Convert patient data to window lists"""
        windows, labels, patient_ids, filenames = [], [], [], []
        
        for patient_id in patient_list:
            data = all_patient_data[patient_id]
            windows.extend(data['windows'])
            labels.extend(data['labels'])
            patient_ids.extend(data['patient_ids'])
            filenames.extend(data['filenames'])
        
        af_count = sum(labels)
        print(f"{split_name}: {len(windows)} windows ({af_count} AF, {len(windows)-af_count} Normal)")
        
        return windows, labels, patient_ids, filenames
    
    def _extract_patient_id(self, filename):
        """Extract unique patient ID"""
        if "non_af" in filename:
            match = re.search(r'non_af_(\d+)', filename)
            return f"HEALTHY_{match.group(1)}" if match else filename
        else:
            match = re.search(r'(?<!non_)af_(\d+)', filename)
            return f"AF_{match.group(1)}" if match else filename


# # Utility functions for backwards compatibility (if needed)
# def load_and_segment_data(dataset_path, val_ratio=0.2, test_ratio=0.2, random_seed=42):
#     """Backwards compatibility function - uses SpectralPreprocessor"""
#     print("Warning: Using deprecated load_and_segment_data. Use SpectralPreprocessor directly.")
#     preprocessor = SpectralPreprocessor()
#     return preprocessor.load_training_data(dataset_path, val_ratio, test_ratio, random_seed)


# if __name__ == '__main__':
#     # Test the preprocessors
#     dataset_path = 'Dataset'
    
#     print("=" * 60)
#     print("Testing SpectralPreprocessor")
#     print("=" * 60)
    
#     spectral_preprocessor = SpectralPreprocessor()
#     train_data, val_data, test_data = spectral_preprocessor.load_training_data(dataset_path)
    
#     train_segments, train_labels, train_patient_ids, _ = train_data
#     print(f"Training segments: {len(train_segments)}")
#     print(f"Sample segment shape: {train_segments[0].shape if train_segments else 'No segments'}")
    
#     print("\n" + "=" * 60)
#     print("Testing TransformerPreprocessor")
#     print("=" * 60)
    
#     transformer_preprocessor = TransformerPreprocessor()
#     train_data, val_data, test_data = transformer_preprocessor.load_training_data(dataset_path)
    
#     train_windows, train_labels, train_patient_ids, _ = train_data
#     print(f"Training windows: {len(train_windows)}")
#     print(f"Sample window shape: {train_windows[0].shape if train_windows else 'No windows'}")

def create_spectral_preprocessor_default():
    """Create SpectralPreprocessor with default training parameters"""
    return SpectralPreprocessor(
        min_fragment_size=375,  # 3 seconds at 125Hz  
        target_fragment_size=1250  # 10 seconds at 125Hz
    )

def create_transformer_preprocessor_default():
    """Create TransformerPreprocessor with default training parameters"""
    return TransformerPreprocessor(
        context_length=500,
        stride=50, 
        sample_rate=50
    )