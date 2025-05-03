"""
Models package for AF Detection
Contains transformer and spectral analysis models
"""

from .transformer import TransformerModel
from .spectral import (
    extract_spectral_entropy,
    extract_frequency_bands,
    extract_wavelet_features,
    extract_spectral_features,
    predict_spectral
)

__all__ = [
    'TransformerModel',
    'extract_spectral_entropy',
    'extract_frequency_bands',
    'extract_wavelet_features',
    'extract_spectral_features',
    'predict_spectral'
]