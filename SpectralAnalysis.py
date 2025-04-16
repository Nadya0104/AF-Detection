import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns

# Define constants
SAMPLING_RATE = 125  # 125 Hz
TARGET_FRAGMENT_SIZE = 1250  # 10 seconds of data
MIN_FRAGMENT_SIZE = 375  # 3 seconds of data (minimum acceptable segment)
DATASET_PATH = "Dataset"  # Path to the dataset folder

def load_and_segment_data(dataset_path, min_fragment_size=MIN_FRAGMENT_SIZE, 
                          target_fragment_size=TARGET_FRAGMENT_SIZE):
    """
    Load all CSV files from the dataset and segment them
    min_fragment_size: Minimum acceptable fragment size (3 seconds = 375 samples)
    target_fragment_size: Target fragment size (10 seconds = 1250 samples)
    """
    X = []  # Feature fragments
    y = []  # Labels (0 for Normal, 1 for AF)
    file_info = []  # To keep track of which file each fragment came from
    segment_lengths = []  # To track the length of each segment for analysis
    
    # Process AF patients (label 1)
    af_path = os.path.join(dataset_path, "AF_Patients")
    if os.path.exists(af_path):
        for filename in os.listdir(af_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(af_path, filename)
                try:
                    data = pd.read_csv(file_path)
                    ppg_data = data['PPG'].values
                    
                    # Find continuous segments without NaN
                    current_segment = []
                    fragment_count = 0
                    
                    for i, value in enumerate(ppg_data):
                        if np.isnan(value):
                            # If we have a segment of sufficient length, add it
                            if len(current_segment) >= min_fragment_size:
                                X.append(np.array(current_segment))
                                y.append(1)  # AF label
                                file_info.append((filename, fragment_count))
                                segment_lengths.append(len(current_segment))
                                fragment_count += 1
                            
                            # Reset segment
                            current_segment = []
                        else:
                            current_segment.append(value)
                            
                            # If we've reached the target size, add segment and start a new one
                            if len(current_segment) == target_fragment_size:
                                X.append(np.array(current_segment))
                                y.append(1)  # AF label
                                file_info.append((filename, fragment_count))
                                segment_lengths.append(len(current_segment))
                                fragment_count += 1
                                current_segment = [] # Reset segment
                    
                    # Add the last segment if it's long enough
                    if len(current_segment) >= min_fragment_size:
                        X.append(np.array(current_segment))
                        y.append(1)  # AF label
                        file_info.append((filename, fragment_count))
                        segment_lengths.append(len(current_segment))
                        fragment_count += 1
                        
                    print(f"Processed AF file: {filename}, created {fragment_count} fragments")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
    
    # Process Healthy patients (label 0)
    healthy_path = os.path.join(dataset_path, "Healthy_Patients")
    if os.path.exists(healthy_path):
        for filename in os.listdir(healthy_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(healthy_path, filename)
                try:
                    data = pd.read_csv(file_path)
                    ppg_data = data['PPG'].values
                    
                    # Find continuous segments without NaN
                    current_segment = []
                    fragment_count = 0
                    
                    for i, value in enumerate(ppg_data):
                        if np.isnan(value):
                            # If we have a segment of sufficient length, add it
                            if len(current_segment) >= min_fragment_size:
                                X.append(np.array(current_segment))
                                y.append(0)  # Normal label
                                file_info.append((filename, fragment_count))
                                segment_lengths.append(len(current_segment))
                                fragment_count += 1
                            
                            # Reset segment
                            current_segment = []
                        else:
                            current_segment.append(value)
                            
                            # If we've reached the target size, add segment and start a new one
                            if len(current_segment) == target_fragment_size:
                                X.append(np.array(current_segment))
                                y.append(0)  # Normal label
                                file_info.append((filename, fragment_count))
                                segment_lengths.append(len(current_segment))
                                fragment_count += 1
                                current_segment = [] # Reset segment
                    
                    # Add the last segment if it's long enough
                    if len(current_segment) >= min_fragment_size:
                        X.append(np.array(current_segment))
                        y.append(0)  # Normal label
                        file_info.append((filename, fragment_count))
                        segment_lengths.append(len(current_segment))
                        fragment_count += 1
                        
                    print(f"Processed Healthy file: {filename}, created {fragment_count} fragments")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
    
    # Log segment length distribution info
    if segment_lengths:
        print(f"Segment length statistics:")
        print(f"  Min: {min(segment_lengths)}, Max: {max(segment_lengths)}")
        print(f"  Mean: {np.mean(segment_lengths):.2f}, Median: {np.median(segment_lengths)}")
        print(f"  Full-length segments (10s): {segment_lengths.count(target_fragment_size)} "
              f"({segment_lengths.count(target_fragment_size)/len(segment_lengths)*100:.1f}%)")
        
        # Create histogram of segment lengths
        plt.figure(figsize=(10, 6))
        plt.hist(segment_lengths, bins=20)
        plt.title('Distribution of Segment Lengths')
        plt.xlabel('Length (samples)')
        plt.ylabel('Count')
        plt.savefig('segment_length_distribution.png')
        plt.close()
    
    return np.array(X, dtype=object), np.array(y)

# def pad_segment(segment, target_length=TARGET_FRAGMENT_SIZE):
#     """
#     Pad a segment to the target length if needed
#     """
#     if len(segment) < target_length:
#         # Use the mean value of the segment for padding
#         mean_val = np.mean(segment)
#         padded_segment = np.pad(segment, (0, target_length - len(segment)), 
#                                'constant', constant_values=mean_val)
#         return padded_segment
#     return segment

# def extract_time_domain_features(ppg_segment):
#     """
#     Extract basic time domain features
#     """
#     mean = np.mean(ppg_segment)
#     std = np.std(ppg_segment)
#     rms = np.sqrt(np.mean(ppg_segment**2))
#     skewness = np.mean((ppg_segment - mean)**3) / (std**3) if std > 0 else 0
#     kurtosis = np.mean((ppg_segment - mean)**4) / (std**4) if std > 0 else 0
    
#     return mean, std, rms, skewness, kurtosis

def extract_spectral_entropy(ppg_segment):
    """
    Extract Spectral Entropy from the frequency domain
    """
    # Apply FFT
    yf = fft(ppg_segment)
    N = len(ppg_segment)
    
    # Get power spectrum (only use positive frequencies)
    power_spectrum = np.abs(yf[:N//2])**2
    
    # Normalize power spectrum for entropy calculation
    total_power = np.sum(power_spectrum)
    if total_power > 0:  # Avoid division by zero
        norm_power_spectrum = power_spectrum / total_power
        
        # Calculate spectral entropy (avoid log(0))
        eps = 1e-10
        spectral_entropy = -np.sum(norm_power_spectrum * np.log2(norm_power_spectrum + eps))
    else:
        spectral_entropy = 0
    
    return spectral_entropy

def extract_frequency_bands(ppg_segment, fs=SAMPLING_RATE):
    """
    Extract energy in different frequency bands
    """
    # Apply FFT
    yf = fft(ppg_segment)
    N = len(ppg_segment)
    xf = fftfreq(N, 1/fs)[:N//2]
    power_spectrum = np.abs(yf[:N//2])**2
    
    # Define frequency bands (in Hz)
    vlf_band = (0.003, 0.04)    # Very low frequency
    lf_band = (0.04, 0.15)      # Low frequency
    hf_band = (0.15, 0.4)       # High frequency
    
    # Calculate energy in each band
    total_power = np.sum(power_spectrum)
    
    vlf_power = np.sum(power_spectrum[(xf >= vlf_band[0]) & (xf < vlf_band[1])]) if total_power > 0 else 0
    lf_power = np.sum(power_spectrum[(xf >= lf_band[0]) & (xf < lf_band[1])]) if total_power > 0 else 0
    hf_power = np.sum(power_spectrum[(xf >= hf_band[0]) & (xf < hf_band[1])]) if total_power > 0 else 0
    
    # Normalize by total power
    if total_power > 0:
        vlf_power_norm = vlf_power / total_power
        lf_power_norm = lf_power / total_power
        hf_power_norm = hf_power / total_power
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    else:
        vlf_power_norm, lf_power_norm, hf_power_norm, lf_hf_ratio = 0, 0, 0, 0
    
    return vlf_power_norm, lf_power_norm, hf_power_norm, lf_hf_ratio

def extract_wavelet_features(ppg_segment, wavelet='db4', level=5):
    """
    Extract features from wavelet decomposition (time-frequency domain)
    """
    # Perform wavelet decomposition
    try:
        coeffs = pywt.wavedec(ppg_segment, wavelet, level=min(level, pywt.dwt_max_level(len(ppg_segment), 
                                                    pywt.Wavelet(wavelet).dec_len)))
        
        # Get approximation coefficients (cA)
        cA = coeffs[0]
        
        # Calculate features
        MAVcA = np.mean(np.abs(cA))  # Mean Absolute Value of approx. coefficients
        AEcA = np.mean(cA**2)        # Average Energy of approx. coefficients
        STDcA = np.std(cA)           # Standard Deviation of approx. coefficients
        
        # Get detail coefficients for last 3 levels
        detail_energy = []
        for i in range(1, min(4, len(coeffs))):
            if i < len(coeffs):
                cD = coeffs[i]
                detail_energy.append(np.mean(cD**2))
            else:
                detail_energy.append(0)
        
        while len(detail_energy) < 3:  # Ensure we always have 3 values
            detail_energy.append(0)
        
        return MAVcA, AEcA, STDcA, detail_energy[0], detail_energy[1], detail_energy[2]
    except Exception as e:
        print(f"Warning: Error in wavelet decomposition: {e}")
        return 0, 0, 0, 0, 0, 0

def extract_features(ppg_segments):
    """
    Extract frequency and time-frequency domain features from each PPG segment
    """
    features = []
    feature_names = [
        'SpEn',           # Spectral Entropy
        'VLF_Power',      # Very Low Frequency normalized power
        'LF_Power',       # Low Frequency normalized power
        'HF_Power',       # High Frequency normalized power
        'LF_HF_Ratio',    # LF/HF power ratio
        # 'Mean',           # Mean of PPG signal
        # 'StdDev',         # Standard deviation of PPG signal
        # 'RMS',            # Root Mean Square of PPG signal
        # 'Skewness',       # Skewness of PPG signal
        # 'Kurtosis',       # Kurtosis of PPG signal
        'MAVcA',          # Mean Absolute Value of approx. wavelet coeffs
        'AEcA',           # Average Energy of approx. wavelet coeffs
        'STDcA',          # Standard Deviation of approx. wavelet coeffs
        'Detail1_Energy', # Energy in detail coefficients level 1
        'Detail2_Energy', # Energy in detail coefficients level 2
        'Detail3_Energy'  # Energy in detail coefficients level 3
    ]
    
    for segment in ppg_segments:
        # Pad segment if necessary
        # padded_segment = pad_segment(segment)
        
        # Extract Spectral Entropy and Frequency band features (frequency domain)
        spectral_entropy = extract_spectral_entropy(segment)
        vlf, lf, hf, lf_hf_ratio = extract_frequency_bands(segment)
        
        # # Extract time domain features
        # mean, std, rms, skewness, kurtosis = extract_time_domain_features(padded_segment)
        
        # Extract Wavelet features (time-frequency domain)
        MAVcA, AEcA, STDcA, d1_energy, d2_energy, d3_energy = extract_wavelet_features(segment)
        
        # Combine features
        feature_vector = [
            spectral_entropy, vlf, lf, hf, lf_hf_ratio,
            # mean, std, rms, skewness, kurtosis,
            MAVcA, AEcA, STDcA, d1_energy, d2_energy, d3_energy
        ]
        features.append(feature_vector)
    
    return np.array(features), feature_names

def visualize_features(X_features, y, feature_names):
    """
    Visualize the distribution of features for AF vs Normal
    """
    X_features_df = pd.DataFrame(X_features, columns=feature_names)
    X_features_df['Label'] = y
    X_features_df['Label'] = X_features_df['Label'].map({0: 'Normal', 1: 'AF'})
    
    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Box plots for each feature
    n_features = len(feature_names)
    n_rows = (n_features + 3) // 4  # 4 features per row
    
    plt.figure(figsize=(20, 5 * n_rows))
    for i, feature in enumerate(feature_names):
        plt.subplot(n_rows, 4, i + 1)
        sns.boxplot(x='Label', y=feature, data=X_features_df)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
    
    plt.savefig('visualizations/feature_distributions.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(14, 12))
    corr_matrix = X_features_df.drop('Label', axis=1).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/feature_correlations.png')
    plt.close()
    
    # Pairplot for selected features (choose 4-5 most important features)
    selected_features = feature_names[:5]  # You can modify this after feature importance is known
    plt.figure(figsize=(15, 15))
    selected_df = X_features_df[selected_features + ['Label']]
    sns.pairplot(selected_df, hue='Label')
    plt.savefig('visualizations/selected_features_pairplot.png')
    plt.close()

def feature_selection(X_train, y_train, X_val, X_test, feature_names):
    """
    Perform feature selection using Random Forest importance
    """
    # Train a Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importances.png')
    plt.close()
    
    # Select top features (e.g., those with importance > mean importance)
    mean_importance = np.mean(importances)
    selected_indices = [i for i, imp in enumerate(importances) if imp > mean_importance]
    
    # Print selected features
    print(f"\nSelected {len(selected_indices)} out of {len(feature_names)} features:")
    for i, idx in enumerate(indices):
        if importances[idx] > mean_importance:
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Create new feature sets with selected features
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_feature_names, selected_indices

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """
    Train and evaluate multiple models, perform hyperparameter tuning,
    and select the best model for AF detection
    """
    # Create directory for model results
    os.makedirs('model_results', exist_ok=True)
    
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
            'gamma': ['scale', 'auto', 0.1],
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
    
    # Select best model based on validation AUC
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
    plt.savefig('model_results/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.plot(fpr, tpr, label=f'AUC = {test_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_model_name}')
    plt.legend()
    plt.savefig('model_results/roc_curve.png')
    plt.close()
    
    # Feature importance (for tree-based models)
    if best_model_name in ['Random Forest', 'XGBoost']:
        if best_model_name == 'Random Forest':
            importances = best_model.feature_importances_
        else:  # XGBoost
            importances = best_model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {best_model_name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('model_results/best_model_feature_importance.png')
        plt.close()
    
    return best_model, scaler


def rfe_feature_selection(X_train, y_train, X_val, X_test, feature_names, 
                         classifier=None, cv=5, step=1):
    """
    Perform feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    X_test : array-like
        Test features
    feature_names : list
        Names of all features
    classifier : sklearn estimator, default=None
        Classifier to use for RFE
        If None, RandomForestClassifier with default parameters is used
    cv : int, default=5
        Number of cross-validation folds
    step : int, default=1
        Number of features to remove at each iteration
        
    Returns:
    --------
    X_train_selected : array-like
        Training data with selected features
    X_val_selected : array-like
        Validation data with selected features
    X_test_selected : array-like
        Test data with selected features
    selected_feature_names : list
        Names of selected features
    selected_indices : list
        Indices of selected features
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Set default classifier if none provided
    if classifier is None:
        classifier = RandomForestClassifier(random_state=42)
    
    print("Performing Recursive Feature Elimination with Cross-Validation (RFECV)...")
    
    # Create RFECV object
    rfecv = RFECV(
        estimator=classifier,
        step=step,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        min_features_to_select=1,
        n_jobs=-1
    )
    
    # Fit on training data
    rfecv.fit(X_train, y_train)
    
    # Get selected features
    selected_indices = np.where(rfecv.support_)[0]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    # Report results
    print(f"\nOptimal number of features: {rfecv.n_features_}")
    print(f"Selected {len(selected_indices)} out of {len(feature_names)} features:")
    
    # Get feature rankings (lower is better)
    feature_ranking = rfecv.ranking_
    
    # Sort selected features by importance for display
    selected_ranking = [(feature_names[i], feature_ranking[i]) for i in selected_indices]
    selected_ranking.sort(key=lambda x: x[1])
    
    for i, (feature, rank) in enumerate(selected_ranking):
        print(f"  {i+1}. {feature} (Rank: {rank})")
    
    # Visualize CV results
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
             rfecv.cv_results_['mean_test_score'], 'o-')
    
    # Add vertical line at optimal number of features
    plt.axvline(x=rfecv.n_features_, color='r', linestyle='--')
    plt.title('Recursive Feature Elimination with Cross-Validation')
    plt.tight_layout()
    plt.savefig('visualizations/rfecv_feature_selection.png')
    plt.close()
    
    # Prepare selected datasets
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_feature_names, selected_indices

def main():
    
    print("Loading and segmenting data...")
    X_segments, y = load_and_segment_data(DATASET_PATH)
    
    print(f"Dataset summary: {len(X_segments)} segments, {sum(y)} AF, {len(X_segments) - sum(y)} Normal")
    
    print("\nExtracting features...")
    X_features, feature_names = extract_features(X_segments)
    
    # # Check for any NaN or inf values in features
    # if np.isnan(X_features).any() or np.isinf(X_features).any():
    #     print("Warning: NaN or Inf values detected in features, replacing with zeros")
    #     X_features = np.nan_to_num(X_features)
    
    # Visualize features
    visualize_features(X_features, y, feature_names)
    
    # Split data into train, validation, and test sets (60/20/20 split)
    # First split into train and temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Then split temporary into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplit sizes - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Perform feature selection
    print("\nPerforming feature selection...")

    # X_train_selected, X_val_selected, X_test_selected, selected_features, selected_indices = feature_selection(
    #     X_train, y_train, X_val, X_test, feature_names
    # )

    X_train_selected, X_val_selected, X_test_selected, selected_features, selected_indices = rfe_feature_selection(
    X_train, y_train, X_val, X_test, feature_names)
    
    # Train and evaluate models with selected features
    print("\nTraining models with selected features...")
    best_model, scaler = train_and_evaluate_models(
        X_train_selected, X_val_selected, X_test_selected, 
        y_train, y_val, y_test, 
        selected_features
    )
    
    # Save the model, scaler, and selected indices for future use
    import joblib
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(best_model, 'saved_models/best_model.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    joblib.dump(selected_indices, 'saved_models/selected_indices.pkl')
    
    # Save feature names for reference
    with open('saved_models/feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Save selected feature names for reference
    with open('saved_models/selected_features.txt', 'w') as f:
        for name in selected_features:
            f.write(f"{name}\n")
    
    print("\nModel, scaler, and feature information saved to 'saved_models' directory")
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()