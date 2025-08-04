"""
Improved Transformer Training Code for AF Detection
Features:
- 5-fold cross-validation
- Patient-based train/validation/test splitting (60/20/20)
- Comprehensive evaluation with confusion matrix, ROC curve, and detailed report
- Prevents data leakage by ensuring patient segments don't cross splits
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import math
from scipy.signal import resample
import joblib
from collections import defaultdict

# Constants
SAMPLING_RATE = 125  # Original sampling rate
TARGET_FRAGMENT_SIZE = 1250  # 10 seconds at 125 Hz
MIN_FRAGMENT_SIZE = 375  # 3 seconds at 125 Hz

# ============= Data Processing Functions =============

def resample_signal(signal, target_rate, original_rate=SAMPLING_RATE):
    """Resample signal from original_rate to target_rate"""
    if target_rate == original_rate:
        return signal
    num_samples = int(len(signal) * target_rate / original_rate)
    return resample(signal, num_samples)

def normalize_signal(signal):
    """Z-score normalization of the signal"""
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std > 0:
        return (signal - signal_mean) / signal_std
    return signal - signal_mean

def create_sliding_windows(signal, window_size, stride=None):
    """Create sliding windows from signal"""
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

def extract_patient_id(filename):
    """Extract unique patient ID that distinguishes between AF and non-AF patients"""
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

def create_segments_from_signal(ppg_data, min_fragment_size, target_fragment_size):
    """Create segments from a single signal"""
    segments = []
    current_segment = []
    
    for value in ppg_data:
        if np.isnan(value):
            if len(current_segment) >= min_fragment_size:
                segments.append(np.array(current_segment))
            current_segment = []
        else:
            current_segment.append(value)
            if len(current_segment) == target_fragment_size:
                segments.append(np.array(current_segment))
                current_segment = []
    
    # Add the last segment if it's long enough
    if len(current_segment) >= min_fragment_size:
        segments.append(np.array(current_segment))
    
    return segments

def load_and_segment_data_for_transformer(dataset_path, 
                                        context_length=500,
                                        stride=50,
                                        sample_rate=50,
                                        val_ratio=0.2, 
                                        test_ratio=0.2, 
                                        random_seed=42):
    """
    Load data and create proper train/validation/test splits ensuring no patient leakage
    Creates windows suitable for transformer training
    """
    import random
    random.seed(random_seed)
    
    # Step 1: Load all data grouped by patients
    all_patient_data = {}  # patient_id -> {'windows': [], 'labels': [], 'filenames': [], 'condition': 'AF'/'Healthy'}
    
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
                
                # Clean signal (remove NaN)
                clean_indices = ~np.isnan(ppg_data)
                if np.any(clean_indices):
                    clean_signal = ppg_data[clean_indices]
                    
                    # Resample to target sampling rate
                    resampled_signal = resample_signal(clean_signal, sample_rate)
                    
                    # Normalize
                    normalized_signal = normalize_signal(resampled_signal)
                    
                    # Create windows for transformer
                    windows = create_sliding_windows(normalized_signal, context_length, stride)
                    
                    if patient_id not in all_patient_data:
                        all_patient_data[patient_id] = {
                            'windows': [], 
                            'labels': [], 
                            'filenames': [], 
                            'condition': 'AF'
                        }
                    
                    all_patient_data[patient_id]['windows'].extend(windows)
                    all_patient_data[patient_id]['labels'].extend([1] * len(windows))  # AF = 1
                    all_patient_data[patient_id]['filenames'].extend([filename] * len(windows))
                    
                    print(f"Processed AF file: {filename}, Patient: {patient_id}, Windows: {len(windows)}")
                
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
                
                # Clean signal (remove NaN)
                clean_indices = ~np.isnan(ppg_data)
                if np.any(clean_indices):
                    clean_signal = ppg_data[clean_indices]
                    
                    # Resample to target sampling rate
                    resampled_signal = resample_signal(clean_signal, sample_rate)
                    
                    # Normalize
                    normalized_signal = normalize_signal(resampled_signal)
                    
                    # Create windows for transformer
                    windows = create_sliding_windows(normalized_signal, context_length, stride)
                    
                    if patient_id not in all_patient_data:
                        all_patient_data[patient_id] = {
                            'windows': [], 
                            'labels': [], 
                            'filenames': [], 
                            'condition': 'Healthy'
                        }
                    
                    all_patient_data[patient_id]['windows'].extend(windows)
                    all_patient_data[patient_id]['labels'].extend([0] * len(windows))  # Healthy = 0
                    all_patient_data[patient_id]['filenames'].extend([filename] * len(windows))
                    
                    print(f"Processed Healthy file: {filename}, Patient: {patient_id}, Windows: {len(windows)}")
                
            except Exception as e:
                print(f"Error processing Healthy file {filename}: {e}")
    
    # Step 2: Determine patient-level labels for stratification
    patients = list(all_patient_data.keys())
    patient_labels = []
    
    for patient_id in patients:
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
    
    # Step 4: Create window-level data for each split
    def create_split_data(patient_list, split_name):
        windows, labels, patient_ids, filenames = [], [], [], []
        af_count = 0
        
        for patient_id in patient_list:
            data = all_patient_data[patient_id]
            windows.extend(data['windows'])
            labels.extend(data['labels'])
            patient_ids.extend([patient_id] * len(data['windows']))
            filenames.extend(data['filenames'])
            
            if data['condition'] == 'AF':
                af_count += 1
        
        healthy_count = len(patient_list) - af_count
        af_windows = sum(labels)
        healthy_windows = len(labels) - af_windows
        
        print(f"{split_name} set:")
        print(f"  Patients: {af_count} AF, {healthy_count} Healthy")
        print(f"  Windows: {af_windows} AF ({af_windows/len(labels)*100:.1f}%), {healthy_windows} Healthy")
        
        return windows, labels, patient_ids, filenames
    
    train_data = create_split_data(train_patients, "Training")
    val_data = create_split_data(val_patients, "Validation")
    test_data = create_split_data(test_patients, "Test")
    
    return train_data, val_data, test_data

def create_train_splits_with_cv(windows, labels, patient_ids, n_folds=5, random_state=42):
    """Create train splits for 5-fold cross-validation based on patient IDs"""
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
        
        # Check class distribution
        train_af = np.sum(labels[train_idx] == 1)
        val_af = np.sum(labels[val_idx] == 1)
        
        print(f"  Train AF: {train_af}/{len(train_idx)} ({train_af/len(train_idx)*100:.1f}%)")
        print(f"  Val AF: {val_af}/{len(val_idx)} ({val_af/len(val_idx)*100:.1f}%)")
        
        fold_counter += 1
    
    return cv_folds

# ============= Dataset Class =============

class PPGTransformerDataset(Dataset):
    """Dataset for transformer training with pre-computed windows"""
    
    def __init__(self, windows, labels, patient_ids=None):
        # Ensure all windows have the same length
        self.context_length = len(windows[0]) if windows else 500
        
        # Filter and pad/truncate windows to ensure consistent size
        processed_windows = []
        processed_labels = []
        processed_patient_ids = []
        
        for i, window in enumerate(windows):
            window = np.array(window, dtype=np.float32)
            
            if len(window) == self.context_length:
                processed_windows.append(window)
            elif len(window) < self.context_length:
                # Pad with zeros
                padded = np.zeros(self.context_length, dtype=np.float32)
                padded[:len(window)] = window
                processed_windows.append(padded)
            else:
                # Truncate
                processed_windows.append(window[:self.context_length])
            
            processed_labels.append(labels[i])
            if patient_ids is not None:
                processed_patient_ids.append(patient_ids[i])
        
        # Convert to tensors
        if processed_windows:
            self.data = torch.tensor(np.stack(processed_windows), dtype=torch.float32)
            self.labels = torch.tensor(processed_labels, dtype=torch.float32)
            self.patient_ids = processed_patient_ids if patient_ids is not None else None
        else:
            self.data = torch.empty((0, self.context_length), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.float32)
            self.patient_ids = []
        
        print(f"Dataset created with {len(self.data)} windows")
        if len(self.labels) > 0:
            af_count = sum(self.labels).item()
            print(f"AF windows: {af_count}, Non-AF windows: {len(self.labels) - af_count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ============= Model Components (same as before) =============

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum
        output = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.output_proj(output)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerModel(nn.Module):
    """Transformer model for AF detection"""
    
    def __init__(self, input_dim=1, d_model=64, num_heads=4, num_layers=4, 
                ff_dim=256, max_seq_len=500, num_classes=1, dropout=0.2):
        super().__init__()
        
        # Initial projection from input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Reshape input if needed [batch, seq_len] -> [batch, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Initial projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global attention pooling
        attn_weights = self.attn_pool(x)
        x = torch.sum(x * attn_weights, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x.squeeze(-1)

# ============= Training and Evaluation Functions =============

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            # Convert outputs to probabilities and predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Handle AUC calculation
    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5
    
    avg_loss = total_loss / len(data_loader.dataset)
    
    return avg_loss, accuracy, precision, recall, f1, auc, all_preds, all_probs, all_targets

def create_visualizations(y_true, y_pred, y_prob, save_dir, prefix="test"):
    """Create confusion matrix and ROC curve"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'AF'],
                yticklabels=['Normal', 'AF'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{prefix.capitalize()} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        plt.plot([0, 1], [0, 1], color='darkorange', lw=2, label='ROC curve (AUC = 0.5)')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix.capitalize()} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_roc_curve.png'))
    plt.close()

def create_detailed_report(test_metrics, patient_metrics, config, save_dir):
    """Create detailed text report"""
    report_path = os.path.join(save_dir, 'detailed_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Transformer Model - Final Test Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write("-" * 20 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("Window-Level Performance (Test Set):\n")
        f.write("-" * 37 + "\n")
        f.write(f"Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {test_metrics['f1']:.4f}\n")
        f.write(f"AUC:       {test_metrics['auc']:.4f}\n\n")
        
        if patient_metrics:
            f.write("Patient-Level Performance (Test Set):\n")
            f.write("-" * 34 + "\n")
            f.write(f"Accuracy:  {patient_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {patient_metrics['precision']:.4f}\n")
            f.write(f"Recall:    {patient_metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {patient_metrics['f1']:.4f}\n")
            if 'auc' in patient_metrics:
                f.write(f"AUC:       {patient_metrics['auc']:.4f}\n")
            f.write("\n")
        
        f.write("Summary:\n")
        f.write("-" * 8 + "\n")
        f.write("This transformer model uses temporal patterns in PPG signals\n")
        f.write("for AF detection. Patient-level splitting ensures no data leakage\n")
        f.write("between training, validation, and test sets. The model was selected\n")
        f.write("based on 5-fold cross-validation performance on the training data.\n")

def train_transformer_model(dataset_path,
                          epochs=30,
                          batch_size=32,
                          learning_rate=1e-4,
                          d_model=64,
                          num_heads=8,
                          num_layers=4,
                          context_length=500,
                          stride=50,
                          sample_rate=50,
                          weight_decay=1e-5,
                          use_scheduler=True,
                          n_folds=5,
                          val_ratio=0.2,
                          test_ratio=0.2,
                          random_seed=42,
                          save_dir='saved_transformer_model'):
    """
    Train transformer model with cross-validation and proper patient-based splitting
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data with proper three-way split
    print("Loading and segmenting data with proper patient-based splits...")
    train_data, val_data, test_data = load_and_segment_data_for_transformer(
        dataset_path=dataset_path,
        context_length=context_length,
        stride=stride,
        sample_rate=sample_rate,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Extract data for each split
    train_windows, train_labels, train_patient_ids, _ = train_data
    val_windows, val_labels, val_patient_ids, _ = val_data
    test_windows, test_labels, test_patient_ids, _ = test_data
    
    # Create datasets
    train_dataset = PPGTransformerDataset(train_windows, train_labels, train_patient_ids)
    val_dataset = PPGTransformerDataset(val_windows, val_labels, val_patient_ids)
    test_dataset = PPGTransformerDataset(test_windows, test_labels, test_patient_ids)
    
    print(f"Data loading complete:")
    print(f"  Training: {len(train_dataset)} windows")
    print(f"  Validation: {len(val_dataset)} windows")
    print(f"  Test: {len(test_dataset)} windows")
    
    # Save configuration
    config = {
        'dataset_path': dataset_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'context_length': context_length,
        'stride': stride,
        'sample_rate': sample_rate,
        'weight_decay': weight_decay,
        'use_scheduler': use_scheduler,
        'n_folds': n_folds,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'n_train_patients': len(set(train_patient_ids)),
        'n_val_patients': len(set(val_patient_ids)),
        'n_test_patients': len(set(test_patient_ids)),
        'n_train_windows': len(train_dataset),
        'n_val_windows': len(val_dataset),
        'n_test_windows': len(test_dataset)
    }
    
    # Save config to file
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create CV folds from training data only
    print("Creating cross-validation folds from training data...")
    cv_folds = create_train_splits_with_cv(
        train_windows, train_labels, train_patient_ids, n_folds=n_folds
    )
    
    # Cross-validation
    fold_scores = []
    fold_models = []
    best_cv_auc = -1
    best_model_state = None
    
    print(f"\nStarting {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, cv_val_idx) in enumerate(cv_folds):
        print(f"\n{'='*80}")
        print(f"Training Fold {fold+1}/{n_folds}")
        print(f"{'='*80}")
        
        # Create fold datasets
        fold_train_dataset = Subset(train_dataset, train_idx)
        fold_val_dataset = Subset(train_dataset, cv_val_idx)
        
        # Create data loaders
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
        
        # Initialize model for this fold
        model = TransformerModel(
            input_dim=1,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=d_model*4,
            max_seq_len=context_length,
            num_classes=1,
            dropout=0.2
        ).to(device)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Training loop for this fold
        best_fold_auc = -1
        best_fold_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss = 0
            
            for batch_x, batch_y in tqdm(fold_train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
            
            avg_train_loss = total_loss / len(fold_train_dataset)
            
            # Validation phase
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, _, _, _ = evaluate_model(
                model, fold_val_loader, criterion, device
            )
            
            # Update learning rate scheduler
            if use_scheduler:
                scheduler.step(val_loss)
            
            # Save best model for this fold
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Store fold results
        fold_scores.append(best_fold_auc)
        if best_fold_model_state:
            model.load_state_dict(best_fold_model_state)
        fold_models.append(model)
        
        # Keep track of globally best model
        if best_fold_auc > best_cv_auc:
            best_cv_auc = best_fold_auc
            best_model_state = best_fold_model_state
        
        print(f"Fold {fold+1} completed. Best validation AUC: {best_fold_auc:.4f}")
    
    # Cross-validation results
    print(f"\n{'='*80}")
    print("Cross-Validation Results")
    print(f"{'='*80}")
    
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"Fold AUC scores: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"Mean CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Select best model
    best_fold_idx = np.argmax(fold_scores)
    best_model = fold_models[best_fold_idx]
    print(f"Selected model from fold {best_fold_idx + 1} (AUC: {fold_scores[best_fold_idx]:.4f})")
    
    # Final evaluation on validation set
    print(f"\n{'='*80}")
    print("Final Evaluation on Validation Set")
    print(f"{'='*80}")
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_preds, val_probs, val_targets = evaluate_model(
        best_model, val_loader, criterion, device
    )
    
    print(f"Validation Results:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  AUC: {val_auc:.4f}")
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc, test_preds, test_probs, test_targets = evaluate_model(
        best_model, test_loader, criterion, device
    )
    
    test_metrics = {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc
    }
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Compute patient-level metrics for test set
    print("\nComputing patient-level metrics...")
    patient_predictions = defaultdict(lambda: {'true_label': None, 'predictions': []})
    
    for i, patient_id in enumerate(test_patient_ids):
        patient_predictions[patient_id]['true_label'] = test_targets[i]
        patient_predictions[patient_id]['predictions'].append(test_probs[i])
    
    # Aggregate patient-level predictions (average probability)
    patient_true = []
    patient_pred_probs = []
    
    for patient_id, data in patient_predictions.items():
        patient_true.append(data['true_label'])
        avg_prob = np.mean(data['predictions'])
        patient_pred_probs.append(avg_prob)
    
    patient_pred_binary = (np.array(patient_pred_probs) > 0.5).astype(int)
    
    # Calculate patient-level metrics
    patient_metrics = {
        'accuracy': accuracy_score(patient_true, patient_pred_binary),
        'precision': precision_score(patient_true, patient_pred_binary, zero_division=0),
        'recall': recall_score(patient_true, patient_pred_binary, zero_division=0),
        'f1': f1_score(patient_true, patient_pred_binary, zero_division=0)
    }
    
    if len(np.unique(patient_true)) > 1:
        patient_metrics['auc'] = roc_auc_score(patient_true, patient_pred_probs)
    
    print(f"Patient-Level Test Results:")
    print(f"  Accuracy: {patient_metrics['accuracy']:.4f}")
    print(f"  Precision: {patient_metrics['precision']:.4f}")
    print(f"  Recall: {patient_metrics['recall']:.4f}")
    print(f"  F1 Score: {patient_metrics['f1']:.4f}")
    if 'auc' in patient_metrics:
        print(f"  AUC: {patient_metrics['auc']:.4f}")
    
    # Save model and results
    print(f"\nSaving results to {save_dir}...")
    
    # Save model
    torch.save(best_model.state_dict(), os.path.join(save_dir, 'transformer_model.pth'))
    
    # Save model configuration for loading
    model_config = {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'context_length': context_length
    }
    torch.save(model_config, os.path.join(save_dir, 'model_config.pth'))
    
    # Save training history and metrics
    results = {
        'cv_scores': fold_scores,
        'mean_cv_auc': mean_auc,
        'std_cv_auc': std_auc,
        'validation_metrics': {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        },
        'test_metrics': test_metrics,
        'patient_metrics': patient_metrics
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save best metrics in format compatible with spectral model
    best_metrics = {
        'validation': {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        },
        'test': test_metrics,
        'patient': patient_metrics
    }
    
    with open(os.path.join(save_dir, 'best_metrics.json'), 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(test_targets, test_preds, test_probs, viz_dir, "test")
    create_visualizations(val_targets, val_preds, val_probs, viz_dir, "validation")
    
    # Create patient-level visualizations
    create_visualizations(patient_true, patient_pred_binary, patient_pred_probs, viz_dir, "patient_test")
    
    # Plot training history (CV scores)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_folds + 1), fold_scores, alpha=0.7)
    plt.axhline(y=mean_auc, color='r', linestyle='--', label=f'Mean AUC: {mean_auc:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Validation AUC')
    plt.title('Cross-Validation AUC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cv_scores.png'))
    plt.close()
    
    # Create detailed report 
    create_detailed_report(test_metrics, patient_metrics, config, save_dir)
    
    print(f"\nTraining completed successfully!")
    print(f"Best model saved to: {save_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"Final test AUC: {test_auc:.4f}")
    
    return best_model

# ============= Main Execution =============

if __name__ == '__main__':
    # Configuration
    config = {
        'dataset_path': 'Dataset',  # Update this path
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'd_model': 64,
        'num_heads': 8,
        'num_layers': 4,
        'context_length': 500,
        'stride': 50,
        'sample_rate': 50,
        'weight_decay': 1e-5,
        'use_scheduler': True,
        'n_folds': 5,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'random_seed': 42,
        'save_dir': 'saved_transformer_model_last'
    }
    
    # Train the model
    trained_model = train_transformer_model(**config)
    
    print("\nTraining pipeline completed successfully!")
    print("Check the saved_transformer_model directory for:")
    print("- transformer_model.pth (model weights)")
    print("- model_config.pth (model architecture)")
    print("- config.json (training configuration)")
    print("- results.json (detailed results)")
    print("- best_metrics.json (metrics in spectral format)")
    print("- detailed_report.txt (comprehensive report)")
    print("- visualizations/ (confusion matrices, ROC curves)")
