"""
Results module for saving model outputs and visualizations
"""

from .model_results import (
    ModelResults,
    save_training_history,
    save_confusion_matrix,
    save_roc_curve,
    save_feature_importance,
    save_model_summary
)

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_distributions,
    plot_feature_correlations,
    plot_spectral_feature_importance,
    plot_segment_length_distribution
)

from .transformer_analysis import (
    analyze_transformer_model,
    evaluate_transformer_on_test,
    create_transformer_report
)

__all__ = [
    'ModelResults',
    'save_training_history',
    'save_confusion_matrix',
    'save_roc_curve',
    'save_feature_importance',
    'save_model_summary',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_distributions',
    'plot_feature_correlations',
    'plot_spectral_feature_importance',
    'plot_segment_length_distribution',
    'analyze_transformer_model',
    'evaluate_transformer_on_test',
    'create_transformer_report'
]