"""
Training script for spectral analysis model
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import itertools


from utils.data_processing import load_and_segment_data, create_train_splits_with_cv
from models.spectral import extract_spectral_features, feature_selection,  FEATURE_NAMES
from results.model_results import ModelResults
from results.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    create_results_report,            
    plot_feature_importance_from_selection,  
)


def extract_features_from_segments(segments):
    """Extract spectral features from segments"""
    features = []
    for segment in segments:
        feature_vector = extract_spectral_features(segment)
        features.append(feature_vector[0])  # extract_spectral_features returns (1, n_features)
    return np.array(features)


def train_spectral_model(dataset_path, save_dir='saved_spectral_model', 
                        val_ratio=0.2, test_ratio=0.2, n_folds=5, random_seed=42):
    """
    Train spectral analysis model with proper train/validation/test splits
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    save_dir : str
        Directory to save results
    val_ratio : float
        Ratio of patients for validation set
    test_ratio : float
        Ratio of patients for test set
    n_folds : int
        Number of folds for cross-validation on training set
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (final_model, scaler)
    """
    # Initialize results tracker
    results = ModelResults('spectral', save_dir)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    print("Loading and segmenting data with proper splits...")
    
    # Load data with proper three-way split
    train_data, val_data, test_data = load_and_segment_data(
        dataset_path, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed
    )
    
    # Extract training data
    train_segments, train_labels, train_patient_ids, train_filenames = train_data
    val_segments, val_labels, val_patient_ids, val_filenames = val_data
    test_segments, test_labels, test_patient_ids, test_filenames = test_data
    
    # Extract features for all sets
    print("Extracting features for all data splits...")
    
    # Training set features
    X_train = extract_features_from_segments(train_segments)
    y_train = np.array(train_labels)
    
    # Validation set features  
    X_val = extract_features_from_segments(val_segments)
    y_val = np.array(val_labels)
    
    # Test set features
    X_test = extract_features_from_segments(test_segments)
    y_test = np.array(test_labels)
    
    print(f"Feature extraction complete:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Save configuration
    results.save_config({
        'dataset_path': dataset_path,
        'min_fragment_size': 375,
        'target_fragment_size': 1250,
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'n_train_patients': len(set(train_patient_ids)),
        'n_val_patients': len(set(val_patient_ids)),
        'n_test_patients': len(set(test_patient_ids)),
        'n_train_segments': len(X_train),
        'n_val_segments': len(X_val),
        'n_test_segments': len(X_test),
        'n_folds': n_folds,
        'random_state': random_seed
    })
    
    # Create CV folds from training data only
    print("Creating cross-validation folds from training data...")
    cv_folds = create_train_splits_with_cv(
        train_segments, y_train, train_patient_ids, n_folds=n_folds
    )
    
    # Define models and parameter grids (same as before)
    models = {
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', 
            random_state=random_seed
        ),
        'SVM': SVC(
            probability=True, 
            class_weight='balanced', 
            random_state=random_seed
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=len(y_train) / sum(y_train),
            random_state=random_seed, 
            eval_metric='logloss'
        )
    }
    
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
    
    # Hyperparameter tuning using CV on training set + validation set for final selection
    print("\nStarting hyperparameter tuning with cross-validation...")
    best_model_configs = {}
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        best_params = None
        best_cv_score = -1
        
        # Try different parameter combinations
        for params in generate_param_combinations(param_grids[model_name]):
            print(f"Parameters: {params}")
            model.set_params(**params)
            
            # Perform cross-validation on training set
            fold_scores = []
            for fold, (train_idx, cv_val_idx) in enumerate(cv_folds):
                # Split training data for this fold
                X_fold_train, X_fold_val = X_train[train_idx], X_train[cv_val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[cv_val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                X_fold_val_scaled = scaler.transform(X_fold_val)
                
                # Train model
                model.fit(X_fold_train_scaled, y_fold_train)
                
                # Evaluate
                if hasattr(model, 'predict_proba'):
                    y_fold_val_prob = model.predict_proba(X_fold_val_scaled)[:, 1]
                    score = roc_auc_score(y_fold_val, y_fold_val_prob)
                else:
                    y_fold_val_pred = model.predict(X_fold_val_scaled)
                    score = accuracy_score(y_fold_val, y_fold_val_pred)
                
                fold_scores.append(score)
                print(f"  Fold {fold+1}/{n_folds}: {score:.4f}")
            
            # Average CV score
            avg_cv_score = np.mean(fold_scores)
            print(f"  Average CV Score: {avg_cv_score:.4f}")
            
            # Update best parameters
            if avg_cv_score > best_cv_score:
                best_cv_score = avg_cv_score
                best_params = params
        
        # Train best model on full training set and evaluate on validation set
        print(f"Training best {model_name} on full training set...")
        model.set_params(**best_params)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        X_val_scaled = scaler.transform(X_val)
        if hasattr(model, 'predict_proba'):
            val_prob = model.predict_proba(X_val_scaled)[:, 1]
            val_score = roc_auc_score(y_val, val_prob)
        else:
            val_pred = model.predict(X_val_scaled)
            val_score = accuracy_score(y_val, val_pred)
        
        print(f"Best {model_name} validation score: {val_score:.4f}")
        
        # Save best configuration
        best_model_configs[model_name] = {
            'params': best_params,
            'cv_score': best_cv_score,
            'val_score': val_score,
            'model': model,
            'scaler': scaler
        }
    
    # Select best model based on validation performance
    best_model_name = max(best_model_configs, key=lambda k: best_model_configs[k]['val_score'])
    best_config = best_model_configs[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_config['cv_score']:.4f}")
    print(f"Best validation score: {best_config['val_score']:.4f}")
    
    print("\nPerforming recursive feature elimination with cross-validation...")
    
    # Use the best model for feature selection (from the selected best model)
    fs_model = models[best_model_name]
    fs_model.set_params(**best_config['params'])

    # Scale features for feature selection (use training data only)
    fs_scaler = StandardScaler()
    X_train_scaled_fs = fs_scaler.fit_transform(X_train)

    print(f"Using {best_model_name} for enhanced feature selection with {n_folds}-fold patient-based cross-validation")
    
    # Use RFE for feature selection on training data only
    selected_indices, selection_scores = feature_selection(
        X_train_scaled_fs,
        y_train,
        train_patient_ids,
        FEATURE_NAMES,
        fs_model,
        target_features=8,
        random_state=random_seed
    )

    print(f"Selected {len(selected_indices)} features: {[FEATURE_NAMES[idx] for idx in selected_indices]}")

    plot_feature_importance_from_selection(
        selected_indices,
        FEATURE_NAMES, 
        save_path=os.path.join(viz_dir, 'feature_importance.png')
    )

    # Save selected indices
    joblib.dump(selected_indices, os.path.join(save_dir, 'selected_indices.pkl'))
    
    # Use only selected features for final training
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"\nTraining final model with {len(selected_indices)} selected features...")
    
    # Train final model on training data with selected features
    final_scaler = StandardScaler()
    X_train_selected_scaled = final_scaler.fit_transform(X_train_selected)
    
    final_model = models[best_model_name]
    final_model.set_params(**best_config['params'])
    final_model.fit(X_train_selected_scaled, y_train)
    
    # Evaluate on training data
    if hasattr(final_model, 'predict_proba'):
        y_train_prob = final_model.predict_proba(X_train_selected_scaled)[:, 1]
        y_train_pred = (y_train_prob > 0.5).astype(int)
    else:
        y_train_pred = final_model.predict(X_train_selected_scaled)
        y_train_prob = y_train_pred
    
    # Calculate training metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'auc': roc_auc_score(y_train, y_train_prob) if hasattr(final_model, 'predict_proba') else 0.5
    }
    
    print("\nFinal model performance (training data):")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate on validation set
    X_val_selected_scaled = final_scaler.transform(X_val_selected)
    
    if hasattr(final_model, 'predict_proba'):
        y_val_prob = final_model.predict_proba(X_val_selected_scaled)[:, 1]
        y_val_pred = (y_val_prob > 0.5).astype(int)
    else:
        y_val_pred = final_model.predict(X_val_selected_scaled)
        y_val_prob = y_val_pred
    
    # Calculate validation metrics
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'auc': roc_auc_score(y_val, y_val_prob) if hasattr(final_model, 'predict_proba') else 0.5
    }
    
    print("\nFinal model performance (validation data):")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate on holdout test set
    X_test_selected_scaled = final_scaler.transform(X_test_selected)
    
    if hasattr(final_model, 'predict_proba'):
        y_test_prob = final_model.predict_proba(X_test_selected_scaled)[:, 1]
        y_test_pred = (y_test_prob > 0.5).astype(int)
    else:
        y_test_pred = final_model.predict(X_test_selected_scaled)
        y_test_prob = y_test_pred
    
    # Calculate test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'auc': roc_auc_score(y_test, y_test_prob) if hasattr(final_model, 'predict_proba') else 0.5
    }
    
    print("\nFinal model performance (holdout test data):")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compute patient-level metrics for test set
    patient_predictions = {}
    
    for i, patient_id in enumerate(test_patient_ids):
        if patient_id not in patient_predictions:
            patient_predictions[patient_id] = {
                'true_label': y_test[i],  # All segments from the same patient have the same label
                'predictions': []
            }
        
        if hasattr(final_model, 'predict_proba'):
            patient_predictions[patient_id]['predictions'].append(y_test_prob[i])
        else:
            patient_predictions[patient_id]['predictions'].append(y_test_pred[i])
    
    # Aggregate patient-level predictions (majority vote or average probability)
    patient_true = []
    patient_pred = []
    
    for patient_id, data in patient_predictions.items():
        patient_true.append(data['true_label'])
        
        if hasattr(final_model, 'predict_proba'):
            # Average probability
            avg_prob = np.mean(data['predictions'])
            patient_pred.append(avg_prob)
        else:
            # Majority vote
            votes = np.array(data['predictions'])
            majority = np.sum(votes) > len(votes) / 2
            patient_pred.append(majority)
    
    # Calculate patient-level metrics
    if hasattr(final_model, 'predict_proba'):
        patient_pred_binary = (np.array(patient_pred) > 0.5).astype(int)
        patient_metrics = {
            'accuracy': accuracy_score(patient_true, patient_pred_binary),
            'precision': precision_score(patient_true, patient_pred_binary),
            'recall': recall_score(patient_true, patient_pred_binary),
            'f1': f1_score(patient_true, patient_pred_binary),
            'auc': roc_auc_score(patient_true, patient_pred)
        }
    else:
        patient_metrics = {
            'accuracy': accuracy_score(patient_true, patient_pred),
            'precision': precision_score(patient_true, patient_pred),
            'recall': recall_score(patient_true, patient_pred),
            'f1': f1_score(patient_true, patient_pred)
        }
    
    print("\nPatient-level metrics (holdout test data):")
    for metric, value in patient_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Update results and save (now includes validation metrics)
    results.update_best_metrics({
        'train': train_metrics,
        'validation': val_metrics,  # Added validation metrics
        'test': test_metrics,
        'patient': patient_metrics
    })

    results.save_metrics()
    
    # Save models and additional data
    joblib.dump(final_scaler, os.path.join(save_dir, 'scaler.pkl'))  # Use final_scaler instead of scaler
    joblib.dump(final_model, os.path.join(save_dir, 'best_model.pkl'))
    joblib.dump(best_model_configs, os.path.join(save_dir, 'cv_results.pkl'))  # Save all model configs, not just cv_results
    
    # Create visualizations for final model (now includes validation set)
    print("\nCreating final visualizations...")
    plot_confusion_matrix(y_test, y_test_pred, os.path.join(viz_dir, 'test_confusion_matrix.png'))
    # plot_confusion_matrix(y_train, y_train_pred, os.path.join(viz_dir, 'train_confusion_matrix.png'))
    # plot_confusion_matrix(y_val, y_val_pred, os.path.join(viz_dir, 'val_confusion_matrix.png'))  # Added validation confusion matrix
    
    if hasattr(final_model, 'predict_proba'):
        plot_roc_curve(y_test, y_test_prob, os.path.join(viz_dir, 'test_roc_curve.png'))
        #plot_roc_curve(y_train, y_train_prob, os.path.join(viz_dir, 'train_roc_curve.png'))
        #plot_roc_curve(y_val, y_val_prob, os.path.join(viz_dir, 'val_roc_curve.png'))  # Added validation ROC curve
    
    # If the model provides feature importances, visualize those too
    # if hasattr(final_model, 'feature_importances_'):
    #     # Save model's feature importance values to file
    #     with open(os.path.join(save_dir, 'model_feature_importances.txt'), 'w') as f:
    #         f.write("Model's Internal Feature Importances:\n")
    #         f.write("====================================\n\n")
            
    #         # Get importances and feature names for selected features only
    #         importances = final_model.feature_importances_
    #         names = [FEATURE_NAMES[i] for i in selected_indices]
            
    #         # Sort by importance
    #         sorted_indices = np.argsort(importances)[::-1]
            
    #         for i in sorted_indices:
    #             f.write(f"{names[i]}: {importances[i]:.6f}\n")
    
    # Create results report
    create_results_report(save_dir, 'spectral')
    
    print(f"\nModel and results saved to {save_dir}")
    print(f"Selected {len(selected_indices)} features: {[FEATURE_NAMES[idx] for idx in selected_indices]}")
    
    # Return final model, scaler, and selected indices
    return final_model, final_scaler, selected_indices


def generate_param_combinations(param_grid):
    """Generate all combinations of parameters from a grid"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    param_combinations = []
    for combination in itertools.product(*values):
        param_dict = {keys[i]: combination[i] for i in range(len(keys))}
        param_combinations.append(param_dict)
    
    return param_combinations


if __name__ == '__main__':
    dataset_path = 'Dataset'
    final_model, scaler, selected_indices = train_spectral_model(dataset_path)