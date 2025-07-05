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
    """Create a comprehensive results report with model metrics"""
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
        
        # Load and report best model metrics (TEST SET ONLY)
        try:
            with open(os.path.join(results_dir, 'best_metrics.json'), 'r') as mf:
                metrics = json.load(mf)
                f.write("Final Model Performance (Test Set):\n")
                f.write("-" * 36 + "\n\n")
                
                # Test metrics only (industry standard)
                if 'test' in metrics:
                    test_metrics = metrics['test']
                    f.write(f"Accuracy:  {test_metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"Precision: {test_metrics.get('precision', 0):.4f}\n")
                    f.write(f"Recall:    {test_metrics.get('recall', 0):.4f}\n")
                    f.write(f"F1 Score:  {test_metrics.get('f1', 0):.4f}\n")
                    f.write(f"AUC:       {test_metrics.get('auc', 0):.4f}\n\n")
                
                # Patient-level metrics (also important for medical applications)
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
                
                # Load CV results to show best model information
                try:
                    import joblib
                    cv_results = joblib.load(os.path.join(results_dir, 'cv_results.pkl'))
                    
                    # Find the best model based on validation score
                    best_model_name = None
                    best_val_score = -1
                    
                    for model_name, config in cv_results.items():
                        if config['val_score'] > best_val_score:
                            best_val_score = config['val_score']
                            best_model_name = model_name
                    
                    if best_model_name:
                        f.write("Best Model Information:\n")
                        f.write("-" * 23 + "\n")
                        f.write(f"Selected Model: {best_model_name}\n")
                        f.write(f"Validation Score: {best_val_score:.4f}\n")
                        f.write(f"Best Parameters: {cv_results[best_model_name]['params']}\n\n")
                
                except (FileNotFoundError, ImportError):
                    f.write("Best model information not available.\n\n")
                
                # Performance interpretation (based on test metrics only)
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
        
        # Load selected features information
        try:
            import joblib
            selected_indices = joblib.load(os.path.join(results_dir, 'selected_indices.pkl'))
            # Import feature names
            from models.spectral import FEATURE_NAMES
            
            f.write("Feature Selection Summary:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Selected {len(selected_indices)} out of {len(FEATURE_NAMES)} features\n")
            f.write("Selected features:\n")
            for i, idx in enumerate(selected_indices):
                f.write(f"  {i+1}. {FEATURE_NAMES[idx]}\n")
            f.write("\n")
            
        except (FileNotFoundError, ImportError):
            f.write("Feature selection information not available.\n\n")
        
        # Summary
        f.write("Summary:\n")
        f.write("-" * 8 + "\n")
        f.write("This model uses spectral feature extraction combined with\n")
        f.write("machine learning classification for AF detection from PPG signals.\n")
        f.write("Patient-level splitting ensures no data leakage between sets.\n")
        f.write("Recursive feature elimination was used to select optimal features.\n")
        f.write("\nFor visualization results, check the 'visualizations' folder.\n")

def plot_feature_selection_scores(scores, feature_counts=None, save_path=None, title="Feature Selection Scores"):
    """
    Plot feature selection scores
    
    Parameters:
    -----------
    scores : list or numpy array
        Scores at each step
    feature_counts : list or numpy array, optional
        Number of features at each step. If None, assumes sequential integers.
    save_path : str, optional
        Path to save plot. If None, displays the plot instead.
    title : str
        Title for the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # If feature_counts is not provided, create sequential integers
    if feature_counts is None:
        feature_counts = range(1, len(scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, scores, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Score')
    plt.title(title)
    plt.grid(True)
    
    # Highlight the optimal number of features
    best_idx = np.argmax(scores)
    plt.axvline(x=feature_counts[best_idx], color='r', linestyle='--')
    plt.text(feature_counts[best_idx] + 0.1, scores[best_idx], 
             f'Optimal: {feature_counts[best_idx]} features', 
             va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance_from_selection(selected_indices, feature_names, save_path):
    """
    Plot feature importance based on selection order
    
    Parameters:
    -----------
    selected_indices : list
        Indices of selected features in order of importance
    feature_names : list
        Names of all features
    save_path : str
        Path to save plot
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


def save_selected_features_list(selected_indices, feature_names, save_path):
    """
    Save list of selected features to text file
    
    Parameters:
    -----------
    selected_indices : list
        Indices of selected features in order of importance
    feature_names : list
        Names of all features
    save_path : str
        Path to save text file
    """
    with open(save_path, 'w') as f:
        f.write("Selected Features for AF Detection\n")
        f.write("=================================\n\n")
        f.write("Features in order of selection (most important first):\n\n")
        
        for i, idx in enumerate(selected_indices):
            f.write(f"{i+1}. {feature_names[idx]}\n")
        
        f.write("\nFeature selection helps reduce overfitting and improves model generalization.\n")
        f.write("The order of selection indicates the relative importance of each feature for AF detection.")


def plot_feature_importances_from_model(model, feature_names, selected_indices=None, save_path=None):
    """
    Plot feature importances from a trained model
    
    Parameters:
    -----------
    model : estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of all features
    selected_indices : list or None
        Indices of selected features (if None, use all features)
    save_path : str or None
        Path to save plot (None means display only)
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    
    if selected_indices is not None:
        # Only show importances for selected features
        names = [feature_names[i] for i in selected_indices]
        importances = importances[selected_indices]
    else:
        names = feature_names
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances from Model')
    plt.barh(range(len(sorted_names)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()