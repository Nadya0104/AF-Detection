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


from utils.data_processing import load_and_segment_data, create_train_val_splits_with_cv
from models.spectral import extract_spectral_features, rfe_feature_selection,  FEATURE_NAMES
from results.model_results import ModelResults
from results.visualization import (
    plot_feature_distributions,
    plot_feature_correlations,
    plot_spectral_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    create_results_report,
    plot_feature_selection_scores,            
    plot_feature_importance_from_selection,  
    save_selected_features_list,              
    plot_feature_importances_from_model
)


def extract_features_from_segments(segments):
    """Extract spectral features from segments"""
    features = []
    for segment in segments:
        feature_vector = extract_spectral_features(segment)
        features.append(feature_vector[0])  # extract_spectral_features returns (1, n_features)
    return np.array(features)


def train_spectral_model(dataset_path, save_dir='saved_spectral_model', test_ratio=0.2, n_folds=5, random_seed=42):
    """
    Train spectral analysis model for AF detection using patient-based train/test split
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    save_dir : str
        Directory to save results
    test_ratio : float
        Ratio of patients to reserve for testing (e.g., 0.2 = 20%)
    n_folds : int
        Number of folds for cross-validation
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (final_model, scaler)
    """
    # Initialize results tracker
    results = ModelResults('spectral', save_dir)
    
    # Create needed directories
    os.makedirs(save_dir, exist_ok=True)
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    print("Loading and segmenting data...")
    # Load data with patient-based splitting
    (train_val_segments, train_val_labels, train_val_patient_ids, train_val_filenames,
     test_segments, test_labels, test_patient_ids, test_filenames) = load_and_segment_data(
        dataset_path, test_ratio=test_ratio, random_seed=random_seed
    )
    
    # Extract features for training segments
    print("Extracting features for training segments...")
    X_train_val_features = []
    y_train_val = np.array(train_val_labels)
    valid_train_indices = []
    
    for i, segment in enumerate(train_val_segments):
        try:
            features = extract_spectral_features(segment)
            X_train_val_features.append(features[0])
            valid_train_indices.append(i)
        except Exception as e:
            print(f"Error extracting features for training segment {i}: {e}")
    
    X_train_val_features = np.array(X_train_val_features)
    
    # Update labels and patient IDs to match valid features
    y_train_val = y_train_val[valid_train_indices]
    train_val_patient_ids = [train_val_patient_ids[i] for i in valid_train_indices]
    
    # Extract features for test segments
    print("Extracting features for test segments...")
    X_test_features = []
    y_test = np.array(test_labels)
    valid_test_indices = []
    
    for i, segment in enumerate(test_segments):
        try:
            features = extract_spectral_features(segment)
            X_test_features.append(features[0])
            valid_test_indices.append(i)
        except Exception as e:
            print(f"Error extracting features for test segment {i}: {e}")
    
    X_test_features = np.array(X_test_features)
    
    # Update test labels to match valid features
    y_test = y_test[valid_test_indices]
    test_patient_ids = [test_patient_ids[i] for i in valid_test_indices]
    
    print(f"Training features shape: {X_train_val_features.shape}")
    print(f"Test features shape: {X_test_features.shape}")
    
    # Save configuration
    results.save_config({
        'dataset_path': dataset_path,
        'min_fragment_size': 375,
        'target_fragment_size': 1250,
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'test_ratio': test_ratio,
        'n_train_patients': len(set(train_val_patient_ids)),
        'n_test_patients': len(set(test_patient_ids)),
        'n_train_segments': len(X_train_val_features),
        'n_test_segments': len(X_test_features),
        'n_folds': n_folds,
        'random_state': random_seed
    })
    
    # Visualize features
    print("Creating feature visualizations...")
    plot_feature_distributions(X_train_val_features, y_train_val, FEATURE_NAMES, viz_dir)
    plot_feature_correlations(X_train_val_features, FEATURE_NAMES,
                            os.path.join(viz_dir, 'feature_correlations.png'))
    
    # Create patient-based CV folds
    print("Creating patient-based cross-validation folds...")
    cv_folds = create_train_val_splits_with_cv(
        [train_val_segments[i] for i in valid_train_indices],
        y_train_val, 
        train_val_patient_ids,
        n_folds=n_folds
    )
    
    # Define models
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
            scale_pos_weight=len(y_train_val) / sum(y_train_val),  # Handle class imbalance
            random_state=random_seed, 
            eval_metric='logloss'
        )
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
    
    # CV training results
    cv_results = {}
    for model_name in models.keys():
        cv_results[model_name] = {
            'params': [],
            'scores': []
        }
    
    # Train and evaluate models with CV
    best_model_configs = {}
    
    print("\nStarting Cross-Validation Training...")
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Track best parameters and score
        best_params = None
        best_cv_score = -1
        
        # Try different parameter combinations
        for params in generate_param_combinations(param_grids[model_name]):
            print(f"Parameters: {params}")
            model.set_params(**params)
            
            # Perform cross-validation
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(cv_folds):
                # Split data for this fold
                X_train, X_val = X_train_val_features[train_idx], X_train_val_features[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                if hasattr(model, 'predict_proba'):
                    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
                    score = roc_auc_score(y_val, y_val_prob)
                else:
                    y_val_pred = model.predict(X_val_scaled)
                    score = accuracy_score(y_val, y_val_pred)
                
                fold_scores.append(score)
                print(f"  Fold {fold+1}/{n_folds}: {score:.4f}")
            
            # Average score across folds
            avg_score = np.mean(fold_scores)
            print(f"  Average CV Score: {avg_score:.4f}")
            
            # Save results
            cv_results[model_name]['params'].append(params)
            cv_results[model_name]['scores'].append(avg_score)
            
            # Update best parameters
            if avg_score > best_cv_score:
                best_cv_score = avg_score
                best_params = params
        
        # Save best configuration
        best_model_configs[model_name] = {
            'params': best_params,
            'score': best_cv_score
        }
        
        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best CV score: {best_cv_score:.4f}")
    
    # Find overall best model
    best_model_name = max(best_model_configs, key=lambda k: best_model_configs[k]['score'])
    best_params = best_model_configs[best_model_name]['params']
    best_cv_score = best_model_configs[best_model_name]['score']
    
    print(f"\nBest model overall: {best_model_name}")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_cv_score:.4f}")
    
    # Perform greedy feature selection with the best model
    print("\nPerforming recursive feature elimination with cross-validation...")
    best_model = models[best_model_name].set_params(**best_params)

    # Use RFE for feature selection
    selected_indices, selection_scores = rfe_feature_selection(
        X_train_val_features, 
        y_train_val, 
        train_val_patient_ids, 
        best_model,
        min_features=1,
        max_features=min(15, X_train_val_features.shape[1]),  # Set a reasonable upper limit
        step=1,  # Remove one feature at a time
        verbose=True
    )

    # Create visualizations for feature selection
    plot_feature_selection_scores(
        selection_scores, 
        feature_counts=range(1, len(selection_scores) + 1),
        save_path=os.path.join(viz_dir, 'feature_selection_scores.png'),
        title="Recursive Feature Elimination CV"
    )

    plot_feature_importance_from_selection(
        selected_indices,
        FEATURE_NAMES, 
        save_path=os.path.join(viz_dir, 'feature_importance.png')
    )

    # Save selected features to text file
    save_selected_features_list(
        selected_indices,
        FEATURE_NAMES,
        save_path=os.path.join(save_dir, 'selected_features.txt')
    )

    # Save selected indices
    joblib.dump(selected_indices, os.path.join(save_dir, 'selected_indices.pkl'))
    
    # Use only selected features for final training
    X_train_val_selected = X_train_val_features[:, selected_indices]
    X_test_selected = X_test_features[:, selected_indices]
    
    print(f"\nTraining final model with {len(selected_indices)} selected features...")
    
    # Train final model on all training data
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val_selected)
    
    final_model = models[best_model_name]
    final_model.set_params(**best_params)
    final_model.fit(X_train_val_scaled, y_train_val)
    
    # Evaluate on training data
    if hasattr(final_model, 'predict_proba'):
        y_train_val_prob = final_model.predict_proba(X_train_val_scaled)[:, 1]
        y_train_val_pred = (y_train_val_prob > 0.5).astype(int)
    else:
        y_train_val_pred = final_model.predict(X_train_val_scaled)
        y_train_val_prob = y_train_val_pred
    
    # Calculate training metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train_val, y_train_val_pred),
        'precision': precision_score(y_train_val, y_train_val_pred),
        'recall': recall_score(y_train_val, y_train_val_pred),
        'f1': f1_score(y_train_val, y_train_val_pred),
        'auc': roc_auc_score(y_train_val, y_train_val_prob) if hasattr(final_model, 'predict_proba') else 0.5
    }
    
    print("\nFinal model performance (training data):")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate on holdout test set
    X_test_scaled = scaler.transform(X_test_selected)
    
    if hasattr(final_model, 'predict_proba'):
        y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = (y_test_prob > 0.5).astype(int)
    else:
        y_test_pred = final_model.predict(X_test_scaled)
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
    
    # Update results and save
    results.update_best_metrics({
        'train': train_metrics,
        'test': test_metrics,
        'patient': patient_metrics
    })
    
    # Save models and additional data
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(final_model, os.path.join(save_dir, 'best_model.pkl'))
    joblib.dump(cv_results, os.path.join(save_dir, 'cv_results.pkl'))
    
    # Create visualizations for final model
    print("\nCreating final visualizations...")
    plot_confusion_matrix(y_test, y_test_pred, os.path.join(viz_dir, 'test_confusion_matrix.png'))
    plot_confusion_matrix(y_train_val, y_train_val_pred, os.path.join(viz_dir, 'train_confusion_matrix.png'))
    
    if hasattr(final_model, 'predict_proba'):
        plot_roc_curve(y_test, y_test_prob, os.path.join(viz_dir, 'test_roc_curve.png'))
        plot_roc_curve(y_train_val, y_train_val_prob, os.path.join(viz_dir, 'train_roc_curve.png'))
    
    # If the model provides feature importances, visualize those too
    if hasattr(final_model, 'feature_importances_'):
        # Save model's feature importance values to file
        with open(os.path.join(save_dir, 'model_feature_importances.txt'), 'w') as f:
            f.write("Model's Internal Feature Importances:\n")
            f.write("====================================\n\n")
            
            # Get importances and feature names for selected features only
            importances = final_model.feature_importances_
            names = [FEATURE_NAMES[i] for i in selected_indices]
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            
            for i in sorted_indices:
                f.write(f"{names[i]}: {importances[i]:.6f}\n")
    
    # Create results report
    create_results_report(save_dir, 'spectral')
    
    print(f"\nModel and results saved to {save_dir}")
    
    return final_model, scaler


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
    final_model, scaler = train_spectral_model(dataset_path)