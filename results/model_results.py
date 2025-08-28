"""
Functions for saving models results and visualizations
"""

import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from models.spectral import FEATURE_NAMES
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ModelResults:
    """Container for model training results"""
    
    def __init__(self, model_type, save_dir):
        self.model_type = model_type  # 'transformer' or 'spectral'
        self.save_dir = save_dir
        self.best_metrics = {}
        self.config = {}
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def update_best_metrics(self, metrics):
        """Update best metrics"""
        self.best_metrics = metrics
    
    def save_config(self, config):
        """Save model configuration"""
        self.config = config
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def save_metrics(self):
        """Save best metrics"""
        metrics_path = os.path.join(self.save_dir, 'best_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=4)



def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['Normal', 'AF']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return fpr, tpr, roc_auc


def create_results_report(results_dir, model_type):
    """Create a comprehensive results report for both spectral and transformer models"""
    report_path = os.path.join(results_dir, 'results_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"Model Results Report - {model_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        
        # Load and report configuration
        try:
            with open(os.path.join(results_dir, 'config.json'), 'r') as cf:
                config = json.load(cf)
                f.write("Model Configuration:\n")
                f.write("-" * 20 + "\n")
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        except FileNotFoundError:
            f.write("Configuration file not found.\n\n")
        
        # Load and report best model metrics 
        try:
            with open(os.path.join(results_dir, 'best_metrics.json'), 'r') as mf:
                metrics = json.load(mf)
                f.write("Final Model Performance (Test Set):\n")
                f.write("-" * 36 + "\n\n")
                
                # Test metrics only 
                if 'test' in metrics:
                    test_metrics = metrics['test']
                    f.write(f"Accuracy:  {test_metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"Precision: {test_metrics.get('precision', 0):.4f}\n")
                    f.write(f"Recall:    {test_metrics.get('recall', 0):.4f}\n")
                    f.write(f"F1 Score:  {test_metrics.get('f1', 0):.4f}\n")
                    f.write(f"AUC:       {test_metrics.get('auc', 0):.4f}\n\n")
                
                # Patient-level metrics 
                if 'patient' in metrics:
                    f.write("Patient-Level Performance (Test Set):\n")
                    f.write("-" * 34 + "\n")
                    patient_metrics = metrics['patient']
                    f.write(f"Accuracy:  {patient_metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"Precision: {patient_metrics.get('precision', 0):.4f}\n")
                    f.write(f"Recall:    {patient_metrics.get('recall', 0):.4f}\n")
                    f.write(f"F1 Score:  {patient_metrics.get('f1', 0):.4f}\n")
                    if 'auc' in patient_metrics:
                        f.write(f"AUC:       {patient_metrics.get('auc', 0):.4f}\n")
                    f.write("\n")
                
                # Model-specific information
                if model_type == 'spectral':
                    # Load CV results to show best model information for spectral
                    try:
                        cv_results = joblib.load(os.path.join(results_dir, 'cv_results.pkl'))
                        
                        # Find the best model based on validation score
                        best_model_name = None
                        best_val_score = -1
                        
                        for model_name, model_config in cv_results.items():
                            if model_config['val_score'] > best_val_score:
                                best_val_score = model_config['val_score']
                                best_model_name = model_name
                        
                        if best_model_name:
                            f.write("Best Model Information:\n")
                            f.write("-" * 23 + "\n")
                            f.write(f"Selected Model: {best_model_name}\n")
                            f.write(f"Validation Score: {best_val_score:.4f}\n")
                            f.write(f"Best Parameters: {cv_results[best_model_name]['params']}\n\n")
                    
                    except (FileNotFoundError, ImportError):
                        f.write("Best model information not available.\n\n")
                
                elif model_type == 'transformer':
                    # Load transformer-specific CV results
                    try:
                        with open(os.path.join(results_dir, 'results.json'), 'r') as rf:
                            transformer_results = json.load(rf)
                            
                            f.write("Cross-Validation Results:\n")
                            f.write("-" * 25 + "\n")
                            f.write(f"Mean CV AUC: {transformer_results.get('mean_cv_auc', 0):.4f} ± {transformer_results.get('std_cv_auc', 0):.4f}\n")
                            if 'cv_scores' in transformer_results:
                                cv_scores = transformer_results['cv_scores']
                                f.write(f"Fold Scores: {[f'{score:.4f}' for score in cv_scores]}\n")
                            f.write("\n")
                    
                    except FileNotFoundError:
                        f.write("Cross-validation results not available.\n\n")
                
                # Performance interpretation 
                if 'test' in metrics:
                    test_acc = metrics['test'].get('accuracy', 0)
                    test_f1 = metrics['test'].get('f1', 0)
                    test_auc = metrics['test'].get('auc', 0)
                    
                    f.write("Performance Interpretation:\n")
                    f.write("-" * 27 + "\n")
                    
                    if test_acc > 0.9:
                        f.write("Excellent: Test accuracy > 90%\n")
                    elif test_acc > 0.8:
                        f.write("Good: Test accuracy > 80%\n")
                    elif test_acc > 0.7:
                        f.write("Fair: Test accuracy > 70%\n")
                    else:
                        f.write("Needs improvement: Test accuracy < 70%\n")
                    
                    if test_f1 > 0.8:
                        f.write("Strong F1 score indicates good balance of precision and recall\n")
                    elif test_f1 > 0.7:
                        f.write("Moderate F1 score\n")
                    else:
                        f.write("Low F1 score indicates precision/recall imbalance\n")
                    
                    if test_auc > 0.9:
                        f.write("Excellent discriminative ability (AUC > 0.9)\n")
                    elif test_auc > 0.8:
                        f.write("Good discriminative ability (AUC > 0.8)\n")
                    elif test_auc > 0.7:
                        f.write("Fair discriminative ability (AUC > 0.7)\n")
                    else:
                        f.write("Poor discriminative ability (AUC < 0.7)\n")
                    
                    f.write("\n")
                        
        except FileNotFoundError:
            f.write("Best model metrics file not found.\n\n")
        
        # Model-specific feature/architecture information
        if model_type == 'spectral':
            # Load selected features information
            try:
                selected_indices = joblib.load(os.path.join(results_dir, 'selected_indices.pkl'))
                
                f.write("Feature Selection Summary:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Selected {len(selected_indices)} out of {len(FEATURE_NAMES)} features\n")
                f.write("Selected features:\n")
                for i, idx in enumerate(selected_indices):
                    f.write(f"  {i+1}. {FEATURE_NAMES[idx]}\n")
                f.write("\n")
                
            except (FileNotFoundError, ImportError):
                f.write("Feature selection information not available.\n\n")
        
        elif model_type == 'transformer':
            # Load transformer architecture information
            try:
                model_config = torch.load(os.path.join(results_dir, 'model_config.pth'), map_location='cpu')
                
                f.write("Model Architecture:\n")
                f.write("-" * 18 + "\n")
                f.write(f"Model Dimension: {model_config.get('d_model', 'N/A')}\n")
                f.write(f"Attention Heads: {model_config.get('num_heads', 'N/A')}\n")
                f.write(f"Transformer Layers: {model_config.get('num_layers', 'N/A')}\n")
                f.write(f"Context Length: {model_config.get('context_length', 'N/A')} samples\n")
                f.write("\n")
                
            except (FileNotFoundError, ImportError):
                f.write("Model architecture information not available.\n\n")
        
        # Summary
        f.write("Summary:\n")
        f.write("-" * 8 + "\n")
        
        if model_type == 'spectral':
            f.write("This model uses spectral feature extraction combined with\n")
            f.write("machine learning classification for AF detection from PPG signals.\n")
            f.write("Patient-level splitting ensures no data leakage between sets.\n")
            f.write("Recursive feature elimination was used to select optimal features.\n")
        
        elif model_type == 'transformer':
            f.write("This transformer model uses temporal patterns in PPG signals\n")
            f.write("for AF detection. Patient-level splitting ensures no data leakage\n")
            f.write("between training, validation, and test sets. The model was selected\n")
            f.write("based on cross-validation performance on the training data.\n")
            f.write("Consistent preprocessing ensures training and inference compatibility.\n")
        
        f.write("\nFor visualization results, check the 'visualizations' folder.\n")


def plot_feature_importance_from_selection(selected_indices, feature_names, save_path):
    """
    Plot feature importance based on selection order
    
    """
    selected_names = [feature_names[i] for i in selected_indices]
    # Reverse order for importance (first selected = most important)
    importance = [(len(selected_indices) - i) for i in range(len(selected_indices))]
    
    # Sort by importance (ascending)
    sorted_idx = np.argsort(importance)
    names = [selected_names[i] for i in sorted_idx]
    importance_values = [importance[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(names)), importance_values, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance Based on Selection Order')
    
    # Add value labels to the bars
    for i, v in enumerate(importance_values):
        plt.text(v + 0.1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()