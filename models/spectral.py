"""
Spectral analysis models and feature extraction for AF detection
"""

import numpy as np
from scipy.fft import fft, fftfreq
import pywt
import joblib
import os


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
    """Extract energy in different frequency bands"""
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


def extract_spectral_features(ppg_segment):
    """Extract all spectral features from a PPG segment"""
    # Extract Spectral Entropy
    spectral_entropy = extract_spectral_entropy(ppg_segment)
    
    # Extract Frequency Band features
    vlf, lf, hf, lf_hf_ratio = extract_frequency_bands(ppg_segment)
    
    # Extract Wavelet features
    MAVcA, AEcA, STDcA, d1_energy, d2_energy, d3_energy = extract_wavelet_features(ppg_segment)
    
    # Combine features
    features = np.array([
        spectral_entropy, vlf, lf, hf, lf_hf_ratio,
        MAVcA, AEcA, STDcA, d1_energy, d2_energy, d3_energy
    ]).reshape(1, -1)
    
    return features


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
    'SpEn',           # Spectral Entropy
    'VLF_Power',      # Very Low Frequency normalized power
    'LF_Power',       # Low Frequency normalized power
    'HF_Power',       # High Frequency normalized power
    'LF_HF_Ratio',    # LF/HF power ratio
    'MAVcA',          # Mean Absolute Value of approx. wavelet coeffs
    'AEcA',           # Average Energy of approx. wavelet coeffs
    'STDcA',          # Standard Deviation of approx. wavelet coeffs
    'Detail1_Energy', # Energy in detail coefficients level 1
    'Detail2_Energy', # Energy in detail coefficients level 2
    'Detail3_Energy'  # Energy in detail coefficients level 3
]