"""
Analysis functions for pre-trained transformer model
"""

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from models.transformer import load_transformer_model
from utils.data_processing import PPGTrainingDataset
from results.visualization import plot_confusion_matrix, plot_roc_curve
from torch.utils.data import DataLoader, Subset


def evaluate_transformer_on_test(model, test_loader, device='cpu'):
    """Evaluate transformer model on test data"""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds),
        'auc': roc_auc_score(all_targets, all_probs)
    }
    
    return metrics, all_targets, all_preds, all_probs


def analyze_transformer_model(model_path, dataset_path, save_dir='transformer_analysis'):
    """Analyze pre-trained transformer model"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_transformer_model(model_path, device)
    
    # Load dataset
    dataset = PPGTrainingDataset(dataset_path, augment=False)
    
    # Create train/test split (if not already split)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        stratify=dataset.labels.numpy(),
        random_state=42
    )
    
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate model
    metrics, y_true, y_pred, y_prob = evaluate_transformer_on_test(model, test_loader, device)
    
    # Save metrics
    import json
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create visualizations
    plot_confusion_matrix(y_true, y_pred, 
                         os.path.join(save_dir, 'confusion_matrix.png'))
    
    plot_roc_curve(y_true, y_prob, 
                  os.path.join(save_dir, 'roc_curve.png'))
    
    # Create report
    create_transformer_report(save_dir, metrics)
    
    return metrics


def create_transformer_report(save_dir, metrics):
    """Create a report for transformer model analysis"""
    report_path = os.path.join(save_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Transformer Model Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write("-" * 25 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        f.write("\nModel Information:\n")
        f.write("-" * 25 + "\n")
        f.write("Architecture: TransformerModel\n")
        f.write("Input: PPG signals (500 samples @ 50 Hz)\n")
        f.write("Output: Binary classification (AF vs Normal)\n")
        
        f.write("\nNotes:\n")
        f.write("-" * 25 + "\n")
        f.write("- Model was trained on Google Colab\n")
        f.write("- Uses sliding window approach for longer signals\n")
        f.write("- Employs multi-head attention mechanism\n")