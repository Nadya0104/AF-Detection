import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns

# Define constants
SAMPLING_RATE = 125  # 125 Hz
FRAGMENT_SIZE = 1250  # 10 seconds of data
MIN_FRAGMENT_SIZE = 625  # Minimum 5 seconds of data
DATASET_PATH = "Dataset"  # Path to the dataset folder

def load_and_segment_data(dataset_path, fragment_size=1250, min_fragment_size=625):
    """
    Load all CSV files from the dataset and segment them into fragments,
    ending segments before NaN values and starting new ones after NaN sequences
    """
    X = []  # Feature fragments
    y = []  # Labels (0 for Normal, 1 for AF)
    file_info = []  # To keep track of which file each fragment came from
    segment_lengths = []  # To track the length of each segment
    
    # Function to process a single file
    def process_file(file_path, label, filename):
        fragments_created = 0
        try:
            data = pd.read_csv(file_path)
            ppg_data = data['PPG'].values
            
            current_segment = []
            
            for i, value in enumerate(ppg_data):
                if np.isnan(value):
                    # End current segment if it's long enough
                    if len(current_segment) >= min_fragment_size:
                        X.append(np.array(current_segment))
                        y.append(label)
                        file_info.append((filename, fragments_created))
                        segment_lengths.append(len(current_segment))
                        fragments_created += 1
                    
                    # Reset segment
                    current_segment = []
                else:
                    # Add value to current segment
                    current_segment.append(value)
                    
                    # Check if segment is full size
                    if len(current_segment) == fragment_size:
                        X.append(np.array(current_segment))
                        y.append(label)
                        file_info.append((filename, fragments_created))
                        segment_lengths.append(len(current_segment))
                        fragments_created += 1
                        current_segment = []
            
            # Add remaining segment if it's long enough
            if len(current_segment) >= min_fragment_size:
                X.append(np.array(current_segment))
                y.append(label)
                file_info.append((filename, fragments_created))
                segment_lengths.append(len(current_segment))
                fragments_created += 1
                
            print(f"Processed file: {filename}, created {fragments_created} fragments")
            return fragments_created
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return 0
    
    # Process AF patients (label 1)
    af_path = os.path.join(dataset_path, "AF_Patients")
    for filename in os.listdir(af_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(af_path, filename)
            process_file(file_path, 1, filename)
    
    # Process Healthy patients (label 0)
    healthy_path = os.path.join(dataset_path, "Healthy_Patients")
    for filename in os.listdir(healthy_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(healthy_path, filename)
            process_file(file_path, 0, filename)
    
    # Print segment length statistics
    if segment_lengths:
        print(f"Segment length statistics:")
        print(f"  Min: {min(segment_lengths)} samples ({min(segment_lengths)/SAMPLING_RATE:.2f} seconds)")
        print(f"  Max: {max(segment_lengths)} samples ({max(segment_lengths)/SAMPLING_RATE:.2f} seconds)")
        print(f"  Mean: {np.mean(segment_lengths):.1f} samples ({np.mean(segment_lengths)/SAMPLING_RATE:.2f} seconds)")
        print(f"  Full-length segments: {segment_lengths.count(fragment_size)} out of {len(segment_lengths)}")
    
    return np.array(X, dtype=object), np.array(y), file_info

def check_for_nans(X_features):
    """Check if feature matrix contains NaN values and report where they occur"""
    nan_rows = np.isnan(X_features).any(axis=1)
    nan_cols = np.isnan(X_features).any(axis=0)
    
    if np.any(nan_rows):
        print(f"WARNING: Found {np.sum(nan_rows)} rows with NaN values")
        
    if np.any(nan_cols):
        feature_names = ['SpEn', 'MAVcA', 'AEcA', 'STDcA']
        nan_feature_names = [feature_names[i] for i, has_nan in enumerate(nan_cols) if has_nan]
        print(f"NaN values found in features: {nan_feature_names}")
        
    return nan_rows, nan_cols

def extract_spectral_entropy(ppg_segment, fs=125):
    """
    Extract Spectral Entropy from the frequency domain with improved robustness
    """
    # Apply FFT
    yf = fft(ppg_segment)
    N = len(ppg_segment)
    
    # Get power spectrum (only use positive frequencies)
    power_spectrum = np.abs(yf[:N//2])**2
    
    # Check for zero total power to avoid division by zero
    total_power = np.sum(power_spectrum)
    if total_power <= 1e-10:
        return 0.0  # Return zero entropy for flat signals
    
    # Normalize power spectrum for entropy calculation
    norm_power_spectrum = power_spectrum / total_power
    
    # Filter out zeros before taking log (avoid log(0))
    non_zero_mask = norm_power_spectrum > 1e-10
    if not np.any(non_zero_mask):
        return 0.0  # If all are effectively zero, return zero entropy
        
    # Calculate spectral entropy only on non-zero elements
    entropy_elements = norm_power_spectrum[non_zero_mask] * np.log2(norm_power_spectrum[non_zero_mask])
    spectral_entropy = -np.sum(entropy_elements)
    
    # Sanity check the result
    if np.isnan(spectral_entropy) or np.isinf(spectral_entropy):
        return 0.0  # Return zero for problematic calculations
        
    return spectral_entropy

def extract_wavelet_features(ppg_segment, wavelet='db4', level=5):
    """
    Extract features from wavelet decomposition with support for variable-length segments
    """
    try:
        # Handle case if segment contains constant values
        if np.std(ppg_segment) < 1e-10:
            return 0.0, 0.0, 0.0
            
        # Calculate appropriate wavelet level based on segment length
        max_level = pywt.dwt_max_level(len(ppg_segment), pywt.Wavelet(wavelet).dec_len)
        actual_level = min(level, max_level)
        
        coeffs = pywt.wavedec(ppg_segment, wavelet, level=actual_level)
        
        # Get approximation coefficients
        cA = coeffs[0]
        
        # Calculate features
        MAVcA = np.mean(np.abs(cA)) if len(cA) > 0 else 0.0
        AEcA = np.mean(cA**2) if len(cA) > 0 else 0.0
        STDcA = np.std(cA) if len(cA) > 1 else 0.0
        
        return MAVcA, AEcA, STDcA
        
    except Exception as e:
        print(f"Error in wavelet decomposition: {e}")
        return 0.0, 0.0, 0.0

def extract_features(ppg_segments):
    """
    Extract frequency and time-frequency domain features from each PPG segment
    with improved error handling
    """
    features = []
    problematic_segments = []
    
    for i, segment in enumerate(ppg_segments):
        try:
            # Check for problematic segments
            if np.any(np.isnan(segment)) or np.any(np.isinf(segment)):
                print(f"Warning: Segment {i} contains NaN or Inf values (should not happen with new segmentation)")
                problematic_segments.append(i)
                features.append([0.0, 0.0, 0.0, 0.0])  # Use zeros for bad segments
                continue
                
            # Extract Spectral Entropy (frequency domain)
            spectral_entropy = extract_spectral_entropy(segment)
            
            # Extract Wavelet features (time-frequency domain)
            MAVcA, AEcA, STDcA = extract_wavelet_features(segment)
            
            # Check for NaN or infinity in results
            feature_vector = [spectral_entropy, MAVcA, AEcA, STDcA]
            if any(np.isnan(val) or np.isinf(val) for val in feature_vector):
                print(f"Warning: NaN or Inf values in features for segment {i}")
                problematic_segments.append(i)
                features.append([0.0, 0.0, 0.0, 0.0])  # Use zeros for bad features
            else:
                features.append(feature_vector)
                
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            problematic_segments.append(i)
            features.append([0.0, 0.0, 0.0, 0.0])  # Use zeros for error cases
    
    if problematic_segments:
        print(f"Found {len(problematic_segments)} problematic segments out of {len(ppg_segments)}")
    
    result = np.array(features)
    # Final check for NaNs
    check_for_nans(result)
    
    return result

def visualize_features(X_features, y, feature_names):
    """
    Visualize the distribution of features for AF vs Normal
    """
    X_features_df = pd.DataFrame(X_features, columns=feature_names)
    X_features_df['Label'] = y
    X_features_df['Label'] = X_features_df['Label'].map({0: 'Normal', 1: 'AF'})
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_names):
        sns.boxplot(x='Label', y=feature, data=X_features_df, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} by Class')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = X_features_df.drop('Label', axis=1).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('feature_correlations.png')
    plt.close()

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate multiple models, perform hyperparameter tuning,
    and select the best model for AF detection
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    # Define parameter grids for GridSearchCV
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    # Train and tune each model
    best_models = {}
    validation_scores = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Perform grid search
            grid_search = GridSearchCV(
                model, param_grids[name],
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_models[name] = best_model
            
            # Evaluate on validation set
            y_val_pred = best_model.predict(X_val_scaled)
            y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]
            
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_prob)
            
            validation_scores[name] = {
                'accuracy': val_accuracy,
                'auc': val_auc,
                'best_params': grid_search.best_params_
            }
            
            print(f"{name} Validation - Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Select best model based on validation AUC
    if validation_scores:
        best_model_name = max(validation_scores, key=lambda x: validation_scores[x]['auc'])
        best_model = best_models[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best validation AUC: {validation_scores[best_model_name]['auc']:.4f}")
        
        # Evaluate best model on test set
        y_test_pred = best_model.predict(X_test_scaled)
        y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_prob)
        
        print(f"\n{best_model_name} Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"AUC: {test_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Feature importance (for tree-based models)
        if best_model_name in ['Random Forest', 'XGBoost']:
            feature_names = ['SpEn', 'MAVcA', 'AEcA', 'STDcA']
            
            if best_model_name == 'Random Forest':
                importances = best_model.feature_importances_
            else:  # XGBoost
                importances = best_model.feature_importances_
            
            # Sort feature importances
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importances - {best_model_name}')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
        
        return best_model, scaler
    else:
        print("No models were successfully trained")
        return None, None

def main():
    """
    Main function to run the entire pipeline with improved error handling
    """
    print("Loading and segmenting data...")
    X_segments, y, file_info = load_and_segment_data(DATASET_PATH, FRAGMENT_SIZE, MIN_FRAGMENT_SIZE)
    
    print(f"Dataset summary: {len(X_segments)} segments, {sum(y)} AF, {len(X_segments) - sum(y)} Normal")
    
    print("\nExtracting features...")
    X_features = extract_features(X_segments)
    
    # Check for and filter out rows with NaN values
    nan_rows, _ = check_for_nans(X_features)
    if np.any(nan_rows):
        # Filter out problematic rows
        X_features_filtered = X_features[~nan_rows]
        y_filtered = y[~nan_rows]
        print(f"Filtered out {np.sum(nan_rows)} problematic rows. Remaining: {len(X_features_filtered)}")
    else:
        X_features_filtered = X_features
        y_filtered = y
    
    # Visualize features
    feature_names = ['SpEn', 'MAVcA', 'AEcA', 'STDcA']
    visualize_features(X_features_filtered, y_filtered, feature_names)
    
    # Split data into train, validation, and test sets (60/20/20 split)
    # First split into train and temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features_filtered, y_filtered, test_size=0.4, random_state=42, stratify=y_filtered
    )
    
    # Then split temporary into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplit sizes - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train and evaluate models
    best_model, scaler = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    if best_model is not None:
        # Save the model and scaler for future use
        import joblib
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        print("\nModel and scaler saved to files: 'best_model.pkl' and 'scaler.pkl'")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()