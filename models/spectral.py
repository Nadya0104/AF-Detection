"""
SpecGroupKFoldral analysis models and feature extraction for AF detection
"""

import numpy as np
from scipy.fft import fft, fftfreq
import scipy.signal as signal
import pywt
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



def extract_spectral_entropy(ppg_segment):
    """Extract Spectral Entropy from the frequency domain"""
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


def extract_frequency_bands(ppg_segment, fs=125):
    # Apply FFT
    yf = fft(ppg_segment)
    N = len(ppg_segment)
    xf = fftfreq(N, 1/fs)[:N//2]
    power_spectrum = np.abs(yf[:N//2])**2
    
    # Define PPG-appropriate frequency bands (in Hz)
    vlf_band = (0.003, 0.04)    # Very low frequency (autonomic)
    lf_band = (0.04, 0.15)      # Low frequency (autonomic modulation)
    resp_band = (0.15, 0.4)     # Respiratory modulation
    cardiac_band = (0.75, 2.5)  # Cardiac frequency (45-150 BPM)
    
    # Calculate energy in each band
    total_power = np.sum(power_spectrum)
    
    if total_power > 0:
        vlf_power = np.sum(power_spectrum[(xf >= vlf_band[0]) & (xf < vlf_band[1])])
        lf_power = np.sum(power_spectrum[(xf >= lf_band[0]) & (xf < lf_band[1])])
        resp_power = np.sum(power_spectrum[(xf >= resp_band[0]) & (xf < resp_band[1])])
        cardiac_power = np.sum(power_spectrum[(xf >= cardiac_band[0]) & (xf < cardiac_band[1])])
        
        # Normalize by total power
        vlf_power_norm = vlf_power / total_power
        lf_power_norm = lf_power / total_power
        resp_power_norm = resp_power / total_power
        cardiac_power_norm = cardiac_power / total_power
        
        # Calculate meaningful ratios for PPG/AF detection
        lf_resp_ratio = lf_power / resp_power if resp_power > 0 else 0
        cardiac_resp_ratio = cardiac_power / resp_power if resp_power > 0 else 0
        
    else:
        vlf_power_norm = lf_power_norm = resp_power_norm = cardiac_power_norm = 0
        lf_resp_ratio = cardiac_resp_ratio = 0
    
    # Return 6 features instead of 3
    return vlf_power_norm, lf_power_norm, resp_power_norm, cardiac_power_norm, lf_resp_ratio, cardiac_resp_ratio


def extract_wavelet_features(ppg_segment, wavelet='db4', level=5):
    """Extract features from wavelet decomposition (time-frequency domain)"""
    try:
        max_level = pywt.dwt_max_level(len(ppg_segment), pywt.Wavelet(wavelet).dec_len)
        coeffs = pywt.wavedec(ppg_segment, wavelet, level=min(level, max_level))
        
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
    
def extract_vfcdm_features(ppg_segment, fs=125):
    """
    Extract Variable Frequency Complex Demodulation (VFCDM) features
    
    Parameters:
    -----------
    ppg_segment : numpy array
        Input PPG signal
    fs : int
        Sampling frequency (Hz)
        
    Returns:
    --------
    numpy array
        VFCDM features (4 features)
    """
    try:
        # Preprocess - detrend and normalize
        ppg_detrended = signal.detrend(ppg_segment)
        ppg_normalized = (ppg_detrended - np.mean(ppg_detrended)) / (np.std(ppg_detrended) if np.std(ppg_detrended) > 0 else 1)
        
        # Use Welch's method to estimate power spectral density
        # This is a simplified approach compared to full VFCDM
        f, Pxx = signal.welch(ppg_normalized, fs=fs, nperseg=min(256, len(ppg_normalized)), scaling='spectrum')
        
        # Define frequency bands for cardiac and respiratory components
        respiratory_band = (0.15, 0.4)  # Respiratory frequency band (Hz)
        cardiac_band = (0.75, 2.5)      # Cardiac frequency band (Hz)
        
        # Extract PSD features
        resp_indices = np.logical_and(f >= respiratory_band[0], f <= respiratory_band[1])
        resp_power = np.sum(Pxx[resp_indices])
        resp_peak_freq = f[resp_indices][np.argmax(Pxx[resp_indices])] if np.any(resp_indices) else 0
        
        cardiac_indices = np.logical_and(f >= cardiac_band[0], f <= cardiac_band[1])
        cardiac_power = np.sum(Pxx[cardiac_indices])
        cardiac_peak_freq = f[cardiac_indices][np.argmax(Pxx[cardiac_indices])] if np.any(cardiac_indices) else 0
        
        # Calculate derived features
        power_ratio = resp_power / cardiac_power if cardiac_power > 0 else 0
        frequency_ratio = resp_peak_freq / cardiac_peak_freq if cardiac_peak_freq > 0 else 0
        
        # Additional features from spectral moments
        # Calculate spectral centroid (weighted average frequency)
        if np.sum(Pxx) > 0:
            centroid = np.sum(f * Pxx) / np.sum(Pxx)
        else:
            centroid = 0
            
        # Spectral flatness (geometric mean / arithmetic mean)
        eps = 1e-10  # Avoid log(0) and division by zero
        if np.all(Pxx > 0) and np.sum(Pxx) > 0:
            flatness = np.exp(np.mean(np.log(Pxx + eps))) / (np.mean(Pxx) + eps)
        else:
            flatness = 0
        
        # Return all VFCDM features
        return np.array([power_ratio, frequency_ratio, centroid, flatness])
        
    except Exception as e:
        print(f"Warning: Error in VFCDM feature extraction: {e}")
        return np.zeros(4)  # Return zeros if there's an error


def extract_spectral_features(ppg_segment):
    """Extract all spectral features from a PPG segment"""
    # Extract Spectral Entropy
    spectral_entropy = extract_spectral_entropy(ppg_segment)
    
    # Extract Frequency Band features
    vlf, lf, resp, cardiac, lf_resp_ratio, cardiac_resp_ratio = extract_frequency_bands(ppg_segment)
    
    # Extract Wavelet features
    MAVcA, AEcA, STDcA, d1_energy, d2_energy, d3_energy = extract_wavelet_features(ppg_segment)
    
    # Extract VFCDM features
    vfcdm_features = extract_vfcdm_features(ppg_segment)
    
    # Combine all features
    features = np.array([
    spectral_entropy, vlf, lf, resp, cardiac, lf_resp_ratio, cardiac_resp_ratio,
    MAVcA, AEcA, STDcA, d1_energy, d2_energy, d3_energy,
    *vfcdm_features  # Unpack VFCDM features
    ]).reshape(1, -1)
    
    return features

# def rfe_feature_selection(X, y, patient_ids, model, min_features=1, max_features=None, step=1, verbose=True):
#     """
#     Perform Recursive Feature Elimination with Cross-Validation
    
#     Parameters:
#     -----------
#     X : numpy array
#         Feature matrix
#     y : numpy array
#         Target labels
#     patient_ids : list or numpy array
#         Patient IDs for each sample
#     model : estimator
#         Model to use for selection
#     min_features : int
#         Minimum number of features to select
#     max_features : int or None
#         Maximum number of features to select (None means X.shape[1])
#     step : int
#         Number of features to remove at each iteration
#     verbose : bool
#         Whether to print progress
    
#     Returns:
#     --------
#     tuple
#         (selected_indices, cv_scores)
#     """
#     from sklearn.feature_selection import RFE, RFECV
#     from sklearn.model_selection import GroupKFold
#     import numpy as np
    
#     if max_features is None:
#         max_features = X.shape[1]
#     else:
#         max_features = min(max_features, X.shape[1])
    
#     # Set up patient-based cross-validation
#     cv = GroupKFold(n_splits=5)
    
#     if verbose:
#         print("Performing Recursive Feature Elimination with Cross-Validation...")
#         print(f"Initial feature set size: {X.shape[1]}")
    
#     # Create RFECV selector
#     selector = RFECV(
#         estimator=model,
#         step=step,
#         cv=list(cv.split(X, y, groups=patient_ids)),  # Pre-defined patient-based splits
#         scoring='roc_auc' if hasattr(model, 'predict_proba') else 'accuracy',
#         min_features_to_select=min_features,
#         n_jobs=-1,  # Use all available cores
#         verbose=1 if verbose else 0
#     )
    
#     # Fit the selector
#     selector.fit(X, y)
    
#     # Get selected features
#     selected_indices = np.where(selector.support_)[0]
    
#     # Limit to max_features if needed
#     if len(selected_indices) > max_features:
#         # Use the ranking to select the top max_features
#         feature_ranks = selector.ranking_
#         top_indices = np.argsort(feature_ranks)[:max_features]
#         selected_indices = np.sort(top_indices)  # Keep in original order
    
#     # Get CV scores
#     cv_scores = selector.cv_results_['mean_test_score']
    
#     if verbose:
#         print(f"Optimal number of features: {len(selected_indices)}")
#         print(f"Selected features: {[FEATURE_NAMES[i] for i in selected_indices]}")
#         # Use cv_results_ instead of grid_scores_
#         best_score = np.max(cv_scores)
#         print(f"Best CV score: {best_score:.4f}")
    
#     return selected_indices, cv_scores


def feature_selection(X_train, y_train, patient_ids, feature_names, 
                     base_model, target_features=8, random_state=42):
    """
    Enhanced feature selection with stability analysis
    """
    from sklearn.feature_selection import RFECV, SelectKBest, f_classif
    from sklearn.model_selection import GroupKFold
    from collections import Counter
    
    print(f"Enhanced feature selection with stability testing...")
    
    # Step 1: Statistical pre-filtering (remove clearly irrelevant features)
    stat_selector = SelectKBest(score_func=f_classif, k=min(12, len(feature_names)))
    X_filtered = stat_selector.fit_transform(X_train, y_train)
    filtered_indices = stat_selector.get_support(indices=True)
    filtered_names = [feature_names[i] for i in filtered_indices]
    
    print(f"Pre-filtered to {len(filtered_names)} statistically significant features")
    
    # Step 2: RFECV with patient-based CV
    cv = GroupKFold(n_splits=5)
    cv_splits = list(cv.split(X_filtered, y_train, groups=patient_ids))
    
    # Enhanced RFECV with better parameters
    rfecv = RFECV(
        estimator=base_model,
        step=1,  # Remove one feature at a time (more thorough)
        cv=cv_splits,  # Patient-based CV
        scoring='roc_auc',  # Good for imbalanced data
        min_features_to_select=max(3, target_features-2),  # Reasonable minimum
        n_jobs=-1,  # Use all cores
        verbose=1
    )
    
    rfecv.fit(X_filtered, y_train)
    
    # Step 3: Stability analysis across multiple seeds
    print("Performing stability analysis...")
    
    feature_stability = Counter()
    seeds = [42, 123, 456, 789, 999]  # Multiple seeds for stability
    
    for seed in seeds:
        # Create model with different seed
        model_copy = base_model.__class__(**base_model.get_params())
        model_copy.set_params(random_state=seed)
        
        # RFECV with this seed
        rfecv_seed = RFECV(
            estimator=model_copy,
            step=1,
            cv=cv_splits,
            scoring='roc_auc',
            min_features_to_select=max(3, target_features-2),
            n_jobs=-1,
            verbose=0
        )
        
        rfecv_seed.fit(X_filtered, y_train)
        
        # Count selected features
        selected_mask = rfecv_seed.support_
        for i, selected in enumerate(selected_mask):
            if selected:
                feature_stability[filtered_names[i]] += 1
    
    # Step 4: Select most stable features
    stability_threshold = len(seeds) * 0.6  # Feature must appear in 60% of runs
    stable_features = [name for name, count in feature_stability.items() 
                      if count >= stability_threshold]
    
    # If not enough stable features, take top ones by stability
    if len(stable_features) < target_features:
        sorted_by_stability = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
        stable_features = [name for name, _ in sorted_by_stability[:target_features]]
    
    # Convert back to original indices
    final_indices = [feature_names.index(name) for name in stable_features]
    
    print(f"Selected {len(stable_features)} stable features:")
    for name in stable_features:
        stability_count = feature_stability[name]
        print(f"  {name}: {stability_count}/{len(seeds)} runs")
    
    # Return in same format as old function
    return final_indices, rfecv.cv_results_['mean_test_score']

def predict_spectral(ppg_segment, model, scaler=None, selected_indices=None):
    """Get prediction from spectral analysis model"""
    # Extract features
    features = extract_spectral_features(ppg_segment)
    
    # Select features if indices are provided
    if selected_indices is not None:
        features = features[:, selected_indices]
    
    # Scale features if scaler is provided
    if scaler is not None:
        features = scaler.transform(features)
    
    # Get prediction
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(features)[0, 1]
    else:
        # For models that don't have predict_proba
        pred = model.predict(features)[0]
        prob = 1.0 if pred > 0.5 else 0.0
    
    return prob


def load_spectral_model(model_path, scaler_path=None, indices_path=None):
    """Load a trained spectral model with its scaler and selected indices"""
    # Load model
    model = joblib.load(model_path)
    
    # Load scaler if provided
    scaler = None
    if scaler_path is not None and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: Could not load scaler from {scaler_path}: {e}")
    
    # Load selected feature indices if provided
    selected_indices = None
    if indices_path is not None and os.path.exists(indices_path):
        try:
            selected_indices = joblib.load(indices_path)
        except Exception as e:
            print(f"Warning: Could not load selected indices from {indices_path}: {e}")
    
    return model, scaler, selected_indices


# Feature names for reference
FEATURE_NAMES = [
    'SpEn',                    # Spectral Entropy
    'VLF_Power',               # Very Low Frequency normalized power
    'LF_Power',                # Low Frequency normalized power  
    'Resp_Power',              # Respiratory band normalized power
    'Cardiac_Power',           # Cardiac band normalized power
    'LF_Resp_Ratio',           # LF/Respiratory power ratio
    'Cardiac_Resp_Ratio',      # Cardiac/Respiratory power ratio
    'MAVcA',                   # Mean Absolute Value of approx. wavelet coeffs
    'AEcA',                    # Average Energy of approx. wavelet coeffs
    'STDcA',                   # Standard Deviation of approx. wavelet coeffs
    'Detail1_Energy',          # Energy in detail coefficients level 1
    'Detail2_Energy',          # Energy in detail coefficients level 2
    'Detail3_Energy',          # Energy in detail coefficients level 3
    'VFCDM_Power_Ratio',       # Respiratory to cardiac power ratio
    'VFCDM_Frequency_Ratio',   # Respiratory to cardiac frequency ratio
    'VFCDM_Centroid',          # Spectral centroid
    'VFCDM_Flatness'           # Spectral flatness
]