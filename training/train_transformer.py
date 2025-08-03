"""
Updated Transformer Training Script - Following Spectral Training Logic
Same data splits, same validation strategy, same metrics as spectral training
With same output saving format as original transformer training
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import from data processing
try:
    from utils.data_processing import (load_and_segment_data, normalize_signal, 
                                       create_sliding_windows, create_train_splits_with_cv)
except ImportError as e:
    print("="*80)
    print(f"ERROR: Could not import from 'utils.data_processing'. {e}")
    print("="*80)
    exit()

class PPGWindowDataset(Dataset):
    """Creates fixed-size, overlapping windows from variable-length PPG segments."""
    def __init__(self, segments, labels, context_length, stride=None):
        self.segments = segments
        self.labels = np.array(labels)
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length // 2
        
        print(f"Creating windows with size {context_length} and stride {self.stride}...")
        self.windows = []
        self.window_labels = []
        for segment, label in tqdm(zip(self.segments, self.labels), total=len(self.segments), desc="Processing Segments into Windows"):
            normalized_segment = normalize_signal(segment)
            segment_windows = create_sliding_windows(normalized_segment, self.context_length, self.stride)
            self.windows.extend(segment_windows)
            self.window_labels.extend([label] * len(segment_windows))
        if not self.windows:
            raise ValueError("No windows were created from the provided segments.")
        self.window_labels = np.array(self.window_labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx]
        return torch.tensor(window, dtype=torch.float32).unsqueeze(-1), torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

class TransformerModel(nn.Module):
    """A standard Transformer-based classifier for sequences."""
    def __init__(self, input_dim, d_model, num_heads, num_layers, max_seq_len):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, src):
        embedded = self.encoder_embedding(src)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=1)
        output = self.classifier(pooled)
        return output

def evaluate(model, data_loader, criterion, device):
    """Evaluates model performance on a given device (GPU or CPU)."""
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_probs = np.array(all_probs).flatten()
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Handle AUC calculation with single class warning
    try:
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.5  # Default AUC when only one class present
            print(f"Warning: Only one class present, setting AUC = 0.5")
    except ValueError:
        auc = 0.5
        print(f"Warning: AUC calculation failed, setting AUC = 0.5")
    
    return avg_loss, accuracy, f1, auc, all_preds, all_probs, all_targets

def generate_final_report(model, test_data, context_length, batch_size, device, save_dir):
    """Generates the final report with images and stats - EXACT FORMAT AS ORIGINAL."""
    print("\nStarting final evaluation on the HELD-OUT test set...")
    test_segments, test_labels, _, _ = test_data
    test_dataset = PPGWindowDataset(test_segments, test_labels, context_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    _, _, _, _, test_preds, test_probs, test_targets = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
    
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    roc_auc = roc_auc_score(test_targets, test_probs)

    print("\n--- Final Test Set Performance ---")
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f} | ROC AUC: {roc_auc:.4f}")

    # Save artifacts - EXACT FORMAT AS ORIGINAL
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_targets, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'AF'], yticklabels=['Healthy', 'AF'])
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Final Test Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'test_confusion_matrix.png')); plt.close()
    
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Final Test ROC Curve')
    plt.legend(loc="lower right"); plt.savefig(os.path.join(save_dir, 'test_roc_curve.png')); plt.close()

    with open(os.path.join(save_dir, 'detailed_report.txt'), 'w') as f:
        f.write("Transformer Model - Final Test Report\n========================================\n\n")
        f.write(f"Metrics Summary:\n- Accuracy:  {accuracy:.4f}\n- Precision: {precision:.4f}\n")
        f.write(f"- Recall:    {recall:.4f}\n- F1-score:  {f1:.4f}\n- ROC AUC:   {roc_auc:.4f}\n\n")
        f.write("Summary:\nThis report details the model's performance on the held-out test set after selecting\n")
        f.write("the best model from a 5-fold cross-validation procedure on the training data.\n")
    print(f"All reports and images saved to '{save_dir}'.")

def main(config):
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Load data with train/val/test splits (SAME AS SPECTRAL)
    print("\nStep 1: Loading data and creating train/validation/test splits...")
    train_data, val_data, test_data = load_and_segment_data(
        dataset_path=config['dataset_path'],
        min_fragment_size=375, target_fragment_size=1250,
        val_ratio=0.2, test_ratio=0.2, random_seed=42
    )
    
    # Step 2: Extract data separately (SAME AS SPECTRAL)
    print("\nStep 2: Extracting training, validation, and test data separately...")
    train_segments, train_labels, train_p_ids, _ = train_data
    val_segments, val_labels, val_p_ids, _ = val_data
    test_segments, test_labels, test_p_ids, _ = test_data
    
    # Create datasets
    train_dataset = PPGWindowDataset(train_segments, train_labels, config['context_length'])
    val_dataset = PPGWindowDataset(val_segments, val_labels, config['context_length'])
    
    # Step 3: 5-fold CV on TRAINING SET ONLY (SAME AS SPECTRAL)
    print("\nStep 3: Creating 5-fold cross-validation splits from TRAINING SET ONLY...")
    cv_folds = create_train_splits_with_cv(
        train_segments, train_labels, train_p_ids, n_folds=5
    )

    fold_scores = []  # Store AUC scores (SAME AS SPECTRAL)
    fold_models = []  # Store models for each fold
    best_cv_auc = -1
    best_model_state = None

    # Step 4: Cross-validation on training set only
    for fold, (train_idx, cv_val_idx) in enumerate(cv_folds):
        print("\n" + "="*80)
        print(f"--- Starting Fold {fold+1}/{len(cv_folds)} ---")
        print("="*80)
        
        # Create fold datasets from training data only
        fold_train_subset = Subset(train_dataset, train_idx)
        fold_val_subset = Subset(train_dataset, cv_val_idx)

        # Weighted sampling for class balance
        train_subset_labels = train_dataset.window_labels[train_idx]
        class_counts = np.bincount(train_subset_labels.astype(int))
        if len(class_counts) > 1:
            class_weights = 1. / class_counts
            sample_weights = np.array([class_weights[int(label)] for label in train_subset_labels])
            sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)
        else:
            sampler = None

        fold_train_loader = DataLoader(fold_train_subset, batch_size=config['batch_size'], sampler=sampler, num_workers=2)
        fold_val_loader = DataLoader(fold_val_subset, batch_size=config['batch_size']*2, shuffle=False, num_workers=2)

        # Initialize model for this fold
        model = TransformerModel(1, config['d_model'], config['num_heads'], config['num_layers'], config['context_length']).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) if config['use_scheduler'] else None

        best_fold_auc = -1
        best_model_state_fold = None

        # Train this fold
        for epoch in range(config['epochs']):
            model.train()
            for batch_x, batch_y in tqdm(fold_train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Train]"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Evaluate on fold validation set - USE AUC (SAME AS SPECTRAL)
            val_loss, _, _, val_auc, _, _, _ = evaluate(model, fold_val_loader, criterion, device)
            if scheduler: 
                scheduler.step(val_loss)
            
            if val_auc > best_fold_auc:
                print(f"Fold {fold+1} Epoch {epoch+1}: New best AUC: {val_auc:.4f}")
                best_fold_auc = val_auc
                best_model_state_fold = {k: v.cpu() for k, v in model.state_dict().items()}

        fold_scores.append(best_fold_auc)
        if best_model_state_fold:
            model.load_state_dict(best_model_state_fold)
        fold_models.append(model)
        
        # Keep track of globally best model
        if best_fold_auc > best_cv_auc:
            best_cv_auc = best_fold_auc
            best_model_state = best_model_state_fold
            
        print(f"--- Fold {fold+1} finished. Best Validation AUC: {best_fold_auc:.4f} ---")

    # Step 5: CV Results Analysis (SAME AS SPECTRAL + ORIGINAL FORMAT)
    print("\n" + "="*80)
    print("--- Cross-Validation Finished ---")
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"5-Fold CV AUC Scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Average Validation AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    best_fold_index = np.argmax(fold_scores)
    best_model = fold_models[best_fold_index]
    print(f"Selecting model from Fold {best_fold_index+1} (AUC: {fold_scores[best_fold_index]:.4f}) for final testing.")

    # Save the best model - EXACT FORMAT AS ORIGINAL
    torch.save(best_model.state_dict(), os.path.join(config['save_dir'], 'transformer_model.pth'))
    model_config = {k: v for k, v in config.items() if k in ['d_model', 'num_heads', 'num_layers', 'context_length']}
    torch.save(model_config, os.path.join(config['save_dir'], 'model_config.pth'))
    print(f"Best model saved to {config['save_dir']}")
    
    # Generate the final report using the held-out test set - EXACT FORMAT AS ORIGINAL
    generate_final_report(best_model, test_data, config['context_length'], config['batch_size'], device, config['save_dir'])
    print("\nProcess completed successfully.")
    
    return best_model

if __name__ == '__main__':
    config = {
        'dataset_path': 'Dataset',
        'save_dir': 'saved_transformer_model',
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'context_length': 500,
        'weight_decay': 1e-5,
        'use_scheduler': True
    }
    main(config)