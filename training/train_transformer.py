"""
Training script for transformer model
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.transformer import TransformerModel
from utils.data_processing import PPGTrainingDataset


def train_transformer_model(dataset_path, 
                          epochs=30, 
                          batch_size=32, 
                          learning_rate=1e-4, 
                          d_model=64,
                          num_heads=4,
                          num_layers=4,
                          context_length=500,
                          weight_decay=1e-5,
                          use_augmentation=True,
                          use_scheduler=True,
                          save_dir='saved_transformer_model'):
    """
    Train transformer model for AF detection
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate
    d_model : int
        Model dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of transformer layers
    context_length : int
        Length of input sequences
    weight_decay : float
        L2 regularization weight
    use_augmentation : bool
        Whether to use data augmentation
    use_scheduler : bool
        Whether to use learning rate scheduler
    save_dir : str
        Directory to save the model
    
    Returns:
    --------
    model : TransformerModel
        Trained model
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    full_dataset = PPGTrainingDataset(
        dataset_path, 
        context_length=context_length, 
        augment=use_augmentation
    )
    
    # Get labels for stratified sampling
    labels = full_dataset.labels.numpy()
    
    # Create stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(np.zeros(len(labels)), labels))
    
    # Create Subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, num_workers=2)
    
    # Initialize model
    model = TransformerModel(
        input_dim=1,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=context_length
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
        
        avg_train_loss = total_loss / len(train_dataset)
        
        # Validation phase
        val_loss, val_accuracy, val_f1 = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate scheduler
        if use_scheduler:
            scheduler.step(val_loss)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
        print("-" * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'transformer_model.pth'))
    
    # Save model configuration
    config = {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'context_length': context_length
    }
    torch.save(config, os.path.join(save_dir, 'model_config.pth'))
    
    print(f"Model saved to {save_dir}")
    return model


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            # Convert outputs to predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return total_loss / len(data_loader.dataset), accuracy, f1


if __name__ == '__main__':
    # Example usage
    config = {
        'dataset_path': 'Dataset',
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 4,
        'context_length': 500,
        'weight_decay': 1e-5,
        'use_augmentation': True,
        'use_scheduler': True
    }
    
    trained_model = train_transformer_model(**config)