"""
Transformer Training Script
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import json
import math
from collections import defaultdict
from utils.data_processing import TransformerPreprocessor
from results.model_results import ModelResults, plot_confusion_matrix, plot_roc_curve, create_results_report


# Model Architecture Components

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




# Dataset and Training Functions

class PPGTransformerDataset(Dataset):
    """Dataset for transformer training - uses preprocessed windows"""
    
    def __init__(self, windows, labels, patient_ids=None):
        if windows:
            # Ensure all windows have the same length
            processed_windows = []
            expected_length = len(windows[0]) if windows else 500
            
            for window in windows:
                window = np.array(window, dtype=np.float32)
                
                if len(window) == expected_length:
                    processed_windows.append(window)
                elif len(window) < expected_length:
                    # Pad with zeros
                    padded = np.zeros(expected_length, dtype=np.float32)
                    padded[:len(window)] = window
                    processed_windows.append(padded)
                else:
                    # Truncate
                    processed_windows.append(window[:expected_length])
            
            self.data = torch.tensor(np.stack(processed_windows), dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.patient_ids = patient_ids
        else:
            self.data = torch.empty((0, 500), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.float32)
            self.patient_ids = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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
    Train transformer model using TransformerPreprocessor 
    """
    
    print("TRANSFORMER MODEL TRAINING")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load and process data with TransformerPreprocessor
    preprocessor = TransformerPreprocessor(
        context_length=context_length,
        stride=stride,
        sample_rate=sample_rate
    )
    
    train_data, val_data, test_data = preprocessor.load_training_data(
        dataset_path=dataset_path,
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
    
    print(f"Data: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Save configuration
    config = {
        'dataset_path': dataset_path,
        'preprocessor': 'TransformerPreprocessor',
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

    preprocessing_config = {
        'model_type': 'transformer',
        'context_length': context_length,
        'stride': stride,
        'sample_rate': sample_rate
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(preprocessing_config, f, indent=4)

    # Create CV folds using preprocessor method
    cv_folds = preprocessor.create_cv_folds(
        train_windows, train_labels, train_patient_ids, n_folds=n_folds
    )
    
    # Cross-validation
    fold_scores = []
    fold_models = []
    best_cv_auc = -1
    best_model_state = None
    
    print(f"Cross-validation ({n_folds} folds):")
    
    for fold, (train_idx, cv_val_idx) in enumerate(cv_folds):
        print(f"Fold {fold+1}/{n_folds}...")
        
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
            
            for batch_x, batch_y in fold_train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
            
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
        
        # Store fold results
        fold_scores.append(best_fold_auc)
        if best_fold_model_state:
            model.load_state_dict(best_fold_model_state)
        fold_models.append(model)
        
        # Keep track of globally best model
        if best_fold_auc > best_cv_auc:
            best_cv_auc = best_fold_auc
            best_model_state = best_fold_model_state
        
        print(f"  Best validation AUC: {best_fold_auc:.4f}")
    
    # CV results
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"CV Results: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Select best model
    best_fold_idx = np.argmax(fold_scores)
    best_model = fold_models[best_fold_idx]
    print(f"Selected model from fold {best_fold_idx + 1}")
    
    # Final evaluation on validation set
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_preds, val_probs, val_targets = evaluate_model(
        best_model, val_loader, criterion, device
    )
    
    val_metrics = {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'auc': val_auc
    }
    
    print(f"Validation: Acc={val_accuracy:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
    
    # Final evaluation on test set
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
    
    print(f"Test: Acc={test_accuracy:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
    
    # Compute patient-level metrics for test set
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
    
    print(f"Patient-level: Acc={patient_metrics['accuracy']:.4f}, F1={patient_metrics['f1']:.4f}, AUC={patient_metrics.get('auc', 'N/A')}")
    
    # Save model and results using existing ModelResults class
    results = ModelResults('transformer', save_dir)
    
    # Save configuration
    results.save_config(config)
    
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
    
    # Update and save metrics
    results.update_best_metrics({
        'validation': val_metrics,
        'test': test_metrics,
        'patient': patient_metrics
    })
    results.save_metrics()
    
    # Save additional transformer-specific data
    additional_results = {
        'cv_scores': fold_scores,
        'mean_cv_auc': mean_auc,
        'std_cv_auc': std_auc
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(additional_results, f, indent=4)
    
    
    plot_confusion_matrix(test_targets, test_preds, os.path.join(viz_dir, 'test_confusion_matrix.png'))
    plot_roc_curve(test_targets, test_probs, os.path.join(viz_dir, 'test_roc_curve.png'))
    create_results_report(save_dir, 'transformer')
    
    print(f"TRAINING COMPLETED! Results saved to: {save_dir}, Test AUC: {test_auc:.4f}")
    
    return best_model, preprocessor


if __name__ == '__main__':
    # Configuration
    config = {
        'dataset_path': 'Dataset',
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
        'save_dir': 'saved_transformer_model_v2'
    }
    
    # Train the model
    trained_model, preprocessor = train_transformer_model(**config)
    
    print("✅ Training completed with consistent preprocessing!")