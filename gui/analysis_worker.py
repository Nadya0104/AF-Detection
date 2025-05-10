"""
Worker thread for PPG signal analysis
"""

from PyQt5.QtCore import QThread, pyqtSignal
from models.transformer import predict_transformer
from models.spectral import  predict_spectral


class AnalysisWorker(QThread):
    """Worker thread for PPG signal analysis"""
    
    progress_update = pyqtSignal(int)
    analysis_complete = pyqtSignal(list, list, int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, ppg_signal, transformer_model, spectral_model, 
                 scaler=None, selected_indices=None, w_transformer=0.5,
                 segment_size=1250, stride=625):
        super().__init__()
        self.ppg_signal = ppg_signal
        self.transformer_model = transformer_model
        self.spectral_model = spectral_model
        self.scaler = scaler
        self.selected_indices = selected_indices
        self.w_transformer = w_transformer
        self.segment_size = segment_size  # Default 10s at 125Hz
        self.stride = stride  # Default 5s overlap
        
    def run(self):
        try:
            # Create segments from the full signal
            segments = []
            segment_indices = []
            
            # Use sliding window approach
            for i in range(0, len(self.ppg_signal) - self.segment_size + 1, self.stride):
                segments.append(self.ppg_signal[i:i + self.segment_size])
                segment_indices.append(i)
            
            # If no segments were created (signal too short)
            if len(segments) == 0:
                # Try to use the whole signal if it's long enough
                if len(self.ppg_signal) >= self.segment_size // 2:  # At least half the desired length
                    segments.append(self.ppg_signal)
                    segment_indices.append(0)
                else:
                    # Signal too short for analysis
                    self.error_occurred.emit("Signal too short for analysis")
                    return
            
            # Analyze each segment
            segment_results = []
            
            for i, segment in enumerate(segments):
                # Update progress
                progress = int((i / len(segments)) * 100)
                self.progress_update.emit(progress)
                
                # Get transformer prediction
                transformer_prob = predict_transformer(
                    segment, 
                    self.transformer_model
                )
                
                # Get spectral prediction
                spectral_prob = predict_spectral(
                    segment,
                    self.spectral_model,
                    self.scaler,
                    self.selected_indices
                )
                
                # Weighted fusion
                final_prob = (self.w_transformer * transformer_prob) + \
                            ((1 - self.w_transformer) * spectral_prob)
                
                # Store result
                segment_results.append(final_prob)
            
            # Complete
            self.progress_update.emit(100)
            self.analysis_complete.emit(segment_results, segment_indices, self.segment_size)
            
        except Exception as e:
            self.error_occurred.emit(str(e))