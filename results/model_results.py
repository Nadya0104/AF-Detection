"""
Classes and functions for saving model results
"""

import os
import json
import numpy as np
import joblib
import torch


class ModelResults:
    """Container for model training results"""
    
    def __init__(self, model_type, save_dir):
        self.model_type = model_type  # 'transformer' or 'spectral'
        self.save_dir = save_dir
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        self.best_metrics = {}
        self.config = {}
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def update_history(self, train_loss, val_metrics):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_precision'].append(val_metrics.get('precision', 0))
        self.history['val_recall'].append(val_metrics.get('recall', 0))
    
    def update_best_metrics(self, metrics):
        """Update best metrics"""
        self.best_metrics = metrics
    
    def save_config(self, config):
        """Save model configuration"""
        self.config = config
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def save_history(self):
        """Save training history"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_metrics(self):
        """Save best metrics"""
        metrics_path = os.path.join(self.save_dir, 'best_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=4)
    
    def save_model(self, model):
        """Save the model (only for spectral, since transformer is already saved)"""
        if self.model_type == 'spectral':
            model_path = os.path.join(self.save_dir, 'best_model.pkl')
            joblib.dump(model, model_path)
    
    def save_all(self, model=None):
        """Save all results"""
        self.save_config(self.config)
        self.save_history()
        self.save_metrics()
        if model is not None and self.model_type == 'spectral':
            self.save_model(model)


def save_training_history(history, save_path):
    """Save training history to file"""
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)


def save_confusion_matrix(cm, save_path):
    """Save confusion matrix data"""
    np.save(save_path, cm)


def save_roc_curve(fpr, tpr, auc_score, save_path):
    """Save ROC curve data"""
    roc_data = {
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        'auc': float(auc_score)
    }
    with open(save_path, 'w') as f:
        json.dump(roc_data, f, indent=4)


def save_feature_importance(feature_names, importances, save_path):
    """Save feature importance scores"""
    importance_data = {
        'features': feature_names,
        'importances': importances.tolist() if hasattr(importances, 'tolist') else importances
    }
    with open(save_path, 'w') as f:
        json.dump(importance_data, f, indent=4)


def save_model_summary(model, save_path):
    """Save model architecture summary"""
    with open(save_path, 'w') as f:
        if hasattr(model, '__class__'):
            f.write(f"Model Architecture: {model.__class__.__name__}\n\n")
        
        if hasattr(model, 'parameters'):
            # PyTorch model
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n\n")
            
            f.write("Layer Details:\n")
            for name, module in model.named_children():
                f.write(f"{name}: {module}\n")
        else:
            # Sklearn model
            f.write(str(model))