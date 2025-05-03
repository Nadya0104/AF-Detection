"""
Training modules for AF Detection models
"""

from .train_transformer import train_transformer_model
from .train_spectral import train_spectral_model

__all__ = [
    'train_transformer_model',
    'train_spectral_model'
]