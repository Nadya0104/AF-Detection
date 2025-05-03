"""
Training script for spectral analysis model
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
from utils.data_processing import load_and_segment_data
from models.spectral import extract_spectral_features, FEATURE_NAMES
from results.model_results import ModelResults
from results.visualization import (
    plot_feature_distributions,
    plot_feature_correlations,
    plot_spectral_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    create_results_report
)


def extract_features_from_segments(segments):
    """Extract spectral features from segments"""
    features = []
    for segment in segments:
        feature_vector = extract_spectral_features(segment)
        features.append(feature_vector[0])  # extract_spectral_features returns (1, n_features)
    return np.array(features)


def train_spectral_model(dataset_path, save_dir='saved_spectral_models'):
    """
    Train spectral analysis model for AF detection
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    best_model : sklearn model
        Trained model
    scaler : StandardScaler
        Feature scaler
    """

    # Initialize results tracker
    results = ModelResults('spectral', save_dir)

    print("Loading and segmenting data...")
    X_segments, y = load_and_segment_data(dataset_path, for_transformer=False)
    
    print("Extracting features...")
    X_features = extract_features_from_segments(X_segments)

    # Save configuration
    results.save_config({
        'dataset_path': dataset_path,
        'min_fragment_size': 375,
        'target_fragment_size': 1250,
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES
    })
    
    # Create visualizations directory
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize features
    plot_feature_distributions(X_features, y, FEATURE_NAMES, viz_dir)
    plot_feature_correlations(X_features, FEATURE_NAMES,
                            os.path.join(viz_dir, 'feature_correlations.png'))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Define parameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1],
            'kernel': ['rbf', 'linear']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    # Train and evaluate models
    best_models = {}
    best_score = -1
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[name],
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        if auc > best_score:
            best_score = auc
            best_model_name = name
    
    # Get best model
    best_model = best_models[best_model_name]
    print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
    
    # Final evaluation on test set
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'auc': roc_auc_score(y_test, y_test_prob)
    }
    
    # Update results
    results.update_best_metrics(test_metrics)
    results.save_all(best_model)
    
    # Save additional spectral-specific items
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(best_model, os.path.join(save_dir, 'best_model.pkl'))
    
    # Create visualizations
    plot_confusion_matrix(y_test, y_test_pred,
                         os.path.join(save_dir, 'confusion_matrix.png'))
    
    plot_roc_curve(y_test, y_test_prob,
                  os.path.join(save_dir, 'roc_curve.png'))
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        plot_spectral_feature_importance(
            FEATURE_NAMES, 
            best_model.feature_importances_,
            os.path.join(save_dir, 'feature_importance.png')
        )
    
    # Create results report
    create_results_report(save_dir, 'spectral')
    
    # Save feature names
    with open(os.path.join(save_dir, 'feature_names.txt'), 'w') as f:
        for name in FEATURE_NAMES:
            f.write(f"{name}\n")
    
    print(f"Model and results saved to {save_dir}")
    return best_model, scaler


if __name__ == '__main__':
    dataset_path = 'Dataset'
    trained_model, scaler = train_spectral_model(dataset_path)