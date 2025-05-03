"""
Script to analyze the pre-trained transformer model
"""

import os
import sys
from results.transformer_analysis import analyze_transformer_model

def main():
    # Paths
    model_path = "saved_transformer_model/transformer_model.pth"
    dataset_path = "Dataset"
    save_dir = "saved_transformer_model/analysis"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Analyze model
    print("Analyzing transformer model...")
    metrics = analyze_transformer_model(model_path, dataset_path, save_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {save_dir}")
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()