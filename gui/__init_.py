"""
GUI modules for AF Detection
"""

from .main_gui import AFDetectionGUI
from .analysis_worker import AnalysisWorker
from .ppg_canvas import PPGCanvas

__all__ = [
    'AFDetectionGUI',
    'AnalysisWorker',
    'PPGCanvas'
]