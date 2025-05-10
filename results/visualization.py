"""
Visualization functions for model results
"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# def plot_training_history(history, save_dir):
#     """Plot training history"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     # Plot loss
#     ax1.plot(history['train_loss'], label='Train Loss')
#     ax1.plot(history['val_loss'], label='Validation Loss')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.set_title('Training and Validation Loss')
#     ax1.legend()
#     ax1.grid(True)
    
#     # Plot metrics
#     ax2.plot(history['val_accuracy'], label='Accuracy')
#     ax2.plot(history['val_f1'], label='F1 Score')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Score')
#     ax2.set_title('Validation Metrics')
#     ax2.legend()
#     ax2.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'training_history.png'))
#     plt.close()


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


def plot_feature_distributions(X_features, y, feature_names, save_dir):
    """Plot distribution of features for AF vs Normal"""
    X_features_df = pd.DataFrame(X_features, columns=feature_names)
    X_features_df['Label'] = y
    X_features_df['Label'] = X_features_df['Label'].map({0: 'Normal', 1: 'AF'})
    
    # Box plots for each feature
    n_features = len(feature_names)
    n_rows = (n_features + 3) // 4  # 4 features per row
    
    plt.figure(figsize=(20, 5 * n_rows))
    for i, feature in enumerate(feature_names):
        plt.subplot(n_rows, 4, i + 1)
        sns.boxplot(x='Label', y=feature, data=X_features_df)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
    plt.close()


def plot_feature_correlations(X_features, feature_names, save_path):
    """Plot feature correlation matrix"""
    X_features_df = pd.DataFrame(X_features, columns=feature_names)
    
    plt.figure(figsize=(14, 12))
    corr_matrix = X_features_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_spectral_feature_importance(feature_names, importances, save_path):
    """Plot feature importance for spectral features"""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# def plot_segment_length_distribution(segment_lengths, save_path):
#     """Plot distribution of segment lengths"""
#     plt.figure(figsize=(10, 6))
#     plt.hist(segment_lengths, bins=20)
#     plt.title('Distribution of Segment Lengths')
#     plt.xlabel('Length (samples)')
#     plt.ylabel('Count')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()


def plot_rfecv_results(n_features_selected, cv_scores, save_path):
    """Plot Recursive Feature Elimination with Cross-Validation results"""
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (AUC)")
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-')
    plt.axvline(x=n_features_selected, color='r', linestyle='--')
    plt.title('Recursive Feature Elimination with Cross-Validation')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_results_report(results_dir, model_type):
    """Create a comprehensive results report"""
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
        
        # Load and report best metrics
        try:
            with open(os.path.join(results_dir, 'best_metrics.json'), 'r') as mf:
                metrics = json.load(mf)
                f.write("Best Performance Metrics:\n")
                f.write("-" * 20 + "\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
        except FileNotFoundError:
            f.write("Metrics file not found.\n\n")
        
        # Training summary
        try:
            with open(os.path.join(results_dir, 'training_history.json'), 'r') as hf:
                history = json.load(hf)
                f.write("Training Summary:\n")
                f.write("-" * 20 + "\n")
                
                # Check if we have training history data
                if history.get('train_loss') and len(history['train_loss']) > 0:
                    f.write(f"Total epochs: {len(history['train_loss'])}\n")
                    f.write(f"Final training loss: {history['train_loss'][-1]:.4f}\n")
                    f.write(f"Final validation loss: {history['val_loss'][-1]:.4f}\n")
                    f.write(f"Best validation accuracy: {max(history['val_accuracy']):.4f}\n")
                    f.write(f"Best validation F1: {max(history['val_f1']):.4f}\n")
                else:
                    f.write("Note: Traditional ML model - no epoch-based training history available\n")
                    f.write("See best_metrics.json for final performance metrics\n")
        except FileNotFoundError:
            f.write("Training history file not found.\n\n")