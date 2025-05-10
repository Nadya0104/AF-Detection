"""
Training script for spectral analysis model
"""

import os
import numpy as np
import itertools 
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
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
    """
    # Initialize results tracker
    results = ModelResults('spectral', save_dir)

    print("Loading and segmenting data...")
    X_segments, y, patient_ids = load_and_segment_data(dataset_path)
    
    print("Extracting features...")
    X_features = extract_features_from_segments(X_segments)
    
    # Convert y to numpy array if it's not already
    y = np.array(y)
    
    # Make patient_ids a numpy array of strings to ensure consistent handling
    patient_ids = np.array([str(pid) for pid in patient_ids])
    
    # Get unique patient IDs
    unique_patients = np.unique(patient_ids)
    print(f"Total unique patients/recordings: {len(unique_patients)}")
    
    # Save configuration
    results.save_config({
        'dataset_path': dataset_path,
        'min_fragment_size': 375,
        'target_fragment_size': 1250,
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'n_patients': len(unique_patients),
        'n_segments': len(X_segments)
    })
    
    # Create visualizations directory
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize features
    plot_feature_distributions(X_features, y, FEATURE_NAMES, viz_dir)
    plot_feature_correlations(X_features, FEATURE_NAMES,
                            os.path.join(viz_dir, 'feature_correlations.png'))
    
    # Split data by patient (grouped) to avoid data leakage
    try:
        # Try to use GroupShuffleSplit
        group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(group_split.split(X_features, y, patient_ids))
        
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        patient_ids_train = patient_ids[train_idx]
        
        print(f"Successfully split data by patient ID")
    except Exception as e:
        print(f"Patient-based split failed: {e}")
        print("Falling back to standard train/test split")
        
        # Fallback to standard train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_features, y, np.arange(len(y)), test_size=0.2, 
            random_state=42, stratify=y
        )
        
        # Keep track of patient IDs
        patient_ids_train = np.array([patient_ids[i] for i in train_idx])
    
    print(f"Training set: {len(X_train)} segments from {len(np.unique(patient_ids_train))} patients/recordings")
    print(f"Test set: {len(X_test)} segments")
    
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
    
    # Setup cross-validation
    try:
        # Try patient-wise cross-validation
        group_cv = GroupKFold(n_splits=5)
        # Do a test split to verify it works
        next(group_cv.split(X_train_scaled, y_train, patient_ids_train))
        print("Using patient-wise cross-validation")
    except Exception as e:
        print(f"Patient-wise cross-validation failed: {e}")
        print("Falling back to standard K-Fold cross-validation")
        
        # Fallback to standard KFold
        from sklearn.model_selection import KFold
        group_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train and evaluate models with safer cross-validation
    best_models = {}
    best_score = -1
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Setup for cross-validation
        best_params = None
        best_cv_score = -1
        
        # Manual grid search with appropriate CV
        for params in generate_param_combinations(param_grids[name]):
            # Set parameters
            model.set_params(**params)
            
            # Cross-validation scores
            cv_scores = []
            
            try:
                if isinstance(group_cv, GroupKFold):
                    # Patient-wise CV
                    for train_cv_idx, val_cv_idx in group_cv.split(X_train_scaled, y_train, patient_ids_train):
                        X_train_cv, X_val_cv = X_train_scaled[train_cv_idx], X_train_scaled[val_cv_idx]
                        y_train_cv, y_val_cv = y_train[train_cv_idx], y_train[val_cv_idx]
                        
                        # Train model
                        model.fit(X_train_cv, y_train_cv)
                        
                        # Evaluate
                        if hasattr(model, 'predict_proba'):
                            y_val_prob = model.predict_proba(X_val_cv)[:, 1]
                            score = roc_auc_score(y_val_cv, y_val_prob)
                        else:
                            y_val_pred = model.predict(X_val_cv)
                            score = accuracy_score(y_val_cv, y_val_pred)
                        
                        cv_scores.append(score)
                else:
                    # Standard KFold CV
                    for train_cv_idx, val_cv_idx in group_cv.split(X_train_scaled):
                        X_train_cv, X_val_cv = X_train_scaled[train_cv_idx], X_train_scaled[val_cv_idx]
                        y_train_cv, y_val_cv = y_train[train_cv_idx], y_train[val_cv_idx]
                        
                        # Train model
                        model.fit(X_train_cv, y_train_cv)
                        
                        # Evaluate
                        if hasattr(model, 'predict_proba'):
                            y_val_prob = model.predict_proba(X_val_cv)[:, 1]
                            score = roc_auc_score(y_val_cv, y_val_prob)
                        else:
                            y_val_pred = model.predict(X_val_cv)
                            score = accuracy_score(y_val_cv, y_val_pred)
                        
                        cv_scores.append(score)
            except Exception as e:
                print(f"Error during cross-validation: {e}")
                # If CV fails, assign a low score
                cv_scores = [0.0]
            
            # Average CV score
            avg_cv_score = np.mean(cv_scores)
            print(f"Params: {params}, CV Score: {avg_cv_score:.4f}")
            
            if avg_cv_score > best_cv_score:
                best_cv_score = avg_cv_score
                best_params = params
        
        # Train final model with best parameters
        if best_params is not None:
            model.set_params(**best_params)
            model.fit(X_train_scaled, y_train)
            best_models[name] = model
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            print(f"Best parameters: {best_params}")
            
            if auc > best_score:
                best_score = auc
                best_model_name = name
    
    # Get best model
    best_model = best_models[best_model_name]
    print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
    
    # Calibrate probabilities on a validation set
    try:
        # Try patient-wise split for calibration
        calib_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        calib_train_idx, calib_idx = next(calib_split.split(X_train_scaled, y_train, patient_ids_train))
        
        X_calib_train = X_train_scaled[calib_train_idx]
        y_calib_train = y_train[calib_train_idx]
        X_calib = X_train_scaled[calib_idx]
        y_calib = y_train[calib_idx]
        
        print("Using patient-wise calibration split")
    except Exception as e:
        print(f"Patient-wise calibration split failed: {e}")
        print("Falling back to random calibration split")
        
        # Fallback to random split
        from sklearn.model_selection import train_test_split
        X_calib_train, X_calib, y_calib_train, y_calib = train_test_split(
            X_train_scaled, y_train, test_size=0.3, random_state=42
        )
    
    # Train a new model on the calibration training set
    calib_model = best_models[best_model_name].__class__(**best_model.get_params())
    calib_model.fit(X_calib_train, y_calib_train)
    
    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(
        estimator=calib_model,
        cv='prefit',  # Model is already fit
        method='isotonic'  # More flexible calibration
    )
    calibrated_model.fit(X_calib, y_calib)
    
    # Final evaluation on test set
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'auc': roc_auc_score(y_test, y_test_prob)
    }
    
    # Update results
    results.update_best_metrics(test_metrics)
    results.save_all(calibrated_model)
    
    # Save additional spectral-specific items
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(calibrated_model, os.path.join(save_dir, 'best_model.pkl'))
    
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
    return calibrated_model, scaler

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
    trained_model, scaler = train_spectral_model(dataset_path)