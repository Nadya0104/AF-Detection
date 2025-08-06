"""
Transformer model for AF detection from PPG signals
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from utils.data_processing import PPGInferenceDataset


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum
        output = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.output_proj(output)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerModel(nn.Module):
    """Transformer model for AF detection from PPG signals"""
    
    def __init__(self, input_dim=1, d_model=64, num_heads=4, num_layers=4, 
                ff_dim=256, max_seq_len=500, num_classes=1, dropout=0.2):
        super().__init__()
        
        # Initial projection from input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Reshape input if needed [batch, seq_len] -> [batch, seq_len, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Initial projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global attention pooling
        attn_weights = self.attn_pool(x)
        x = torch.sum(x * attn_weights, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x.squeeze(-1)
    

def load_transformer_model(model_path, device='cpu'):
    """Load a trained transformer model"""
    model = TransformerModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_transformer(ppg_signal, model, device='cpu', batch_size=32, 
                       model_dir='saved_transformer_model'):
    """Get prediction from transformer model with training-consistent parameters"""
    
    # Load training configuration for consistency
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        context_length = config.get('context_length', 500)
        stride = config.get('stride', 50)
        sample_rate = config.get('sample_rate', 50)
        
        print(f"Using training config: context_length={context_length}, stride={stride}, sample_rate={sample_rate}")
    else:
        # Fallback to training defaults
        context_length, stride, sample_rate = 500, 50, 50
        print(f"Config not found, using defaults: context_length={context_length}, stride={stride}, sample_rate={sample_rate}")
    
    # Clean signal first (match training preprocessing)
    clean_indices = ~np.isnan(ppg_signal)
    if np.any(clean_indices):
        clean_signal = ppg_signal[clean_indices]
    else:
        print("Warning: Signal contains only NaN values, returning default prediction")
        return 0.5
    
    # Create dataset with training-consistent parameters
    dataset = PPGInferenceDataset(
        [clean_signal], 
        context_length=context_length, 
        sample_rate=sample_rate,
        stride=stride  # Now matches training
    )
    
    # If dataset is empty, return default prediction
    if len(dataset) == 0:
        print("Warning: Dataset is empty after preprocessing, returning default prediction")
        return 0.5
    
    # Create dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    window_probs = []
    
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            window_probs.extend(probs)
    
    # Average predictions from all windows
    avg_prob = np.mean(window_probs)
    
    print(f"Processed {len(window_probs)} windows, average probability: {avg_prob:.4f}")
    
    return avg_prob