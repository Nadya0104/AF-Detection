"""
Main GUI window for AF Detection application
"""

import os
import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QWidget, QScrollArea, 
                            QFrame, QProgressBar, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from models.transformer import load_transformer_model
from models.spectral import load_spectral_model
from .ppg_canvas import PPGCanvas
from .analysis_worker import AnalysisWorker


class AFDetectionGUI(QMainWindow):
    """Main GUI window for AF detection application"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("AF Detection from PPG Signal")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize models and data
        self.transformer_model = None
        self.spectral_model = None
        self.scaler = None
        self.selected_indices = None
        self.w_transformer = 0.5
        
        self.ppg_signal = None
        self.current_filename = None
        
        # Create UI elements
        self.init_ui()
        
        # Try to load models
        self.load_models()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create top control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # File selection button
        self.load_button = QPushButton("Load PPG Data")
        self.load_button.clicked.connect(self.load_ppg_file)
        control_layout.addWidget(self.load_button)
        
        # Add a gap
        spacer = QWidget()
        spacer.setMinimumWidth(20)
        control_layout.addWidget(spacer)
        
        # Add model weight slider or dropdown
        weight_label = QLabel("Model Weight:")
        control_layout.addWidget(weight_label)
        
        self.weight_combo = QComboBox()
        weight_options = ["50/50 Blend", "Favor Transformer", "Favor Spectral", 
                         "Transformer Only", "Spectral Only"]
        self.weight_combo.addItems(weight_options)
        self.weight_combo.currentIndexChanged.connect(self.update_model_weight)
        control_layout.addWidget(self.weight_combo)
        
        # Add a gap
        spacer2 = QWidget()
        spacer2.setMinimumWidth(20)
        control_layout.addWidget(spacer2)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze Signal")
        self.analyze_button.clicked.connect(self.analyze_signal)
        self.analyze_button.setEnabled(False)
        control_layout.addWidget(self.analyze_button)
        
        # Export results button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        control_layout.addWidget(self.export_button)
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_display)
        self.clear_button.setEnabled(False)
        control_layout.addWidget(self.clear_button)
        
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.status_label)
        
        # Create a horizontal divider line
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(h_line)
        
        # Create matplotlib canvas for PPG signal visualization
        self.ppg_canvas = PPGCanvas(self, width=10, height=6)
        main_layout.addWidget(self.ppg_canvas)
        
        # Add results panel
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_layout = QHBoxLayout()
        
        # Left side: Summary statistics
        self.summary_label = QLabel("No results available")
        self.summary_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_label.setFont(QFont("Arial", 10))
        self.summary_label.setWordWrap(True)
        results_layout.addWidget(self.summary_label)
        
        # Right side: Detailed segments
        self.segments_label = QLabel("Load a PPG file and analyze to see results")
        self.segments_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.segments_label.setFont(QFont("Arial", 10))
        self.segments_label.setWordWrap(True)
        
        # Add scroll area for segments
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.segments_label)
        scroll_area.setWidgetResizable(True)
        results_layout.addWidget(scroll_area)
        
        results_frame.setLayout(results_layout)
        main_layout.addWidget(results_frame)
        
        # Set the main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def load_models(self):
        """Load the transformer and spectral models"""
        try:
            # Check for model files
            transformer_path = "saved_transformer_model/transformer_model.pth"
            spectral_path = "saved_spectral_models/best_model.pkl"
            scaler_path = "saved_spectral_models/scaler.pkl"
            indices_path = "saved_spectral_models/selected_indices.pkl"
            
            # Load transformer model
            self.transformer_model = load_transformer_model(transformer_path, 'cpu')
            
            # Load spectral model with scaler and selected indices
            self.spectral_model, self.scaler, self.selected_indices = load_spectral_model(
                spectral_path, scaler_path, indices_path
            )
            
            self.status_label.setText("Models loaded successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error loading models: {str(e)}")
            QMessageBox.critical(self, "Model Loading Error", 
                                f"Could not load required models: {str(e)}")
    
    def update_model_weight(self):
        """Update the weight between models based on user selection"""
        selected_option = self.weight_combo.currentText()
        
        if selected_option == "50/50 Blend":
            self.w_transformer = 0.5
        elif selected_option == "Favor Transformer":
            self.w_transformer = 0.7
        elif selected_option == "Favor Spectral":
            self.w_transformer = 0.3
        elif selected_option == "Transformer Only":
            self.w_transformer = 1.0
        elif selected_option == "Spectral Only":
            self.w_transformer = 0.0
            
        self.status_label.setText(
            f"Model weight updated: Transformer {self.w_transformer:.1f} / "
            f"Spectral {1-self.w_transformer:.1f}"
        )
    
    def load_ppg_file(self):
        """Load a PPG signal from a CSV file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open PPG Data File", "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Check if 'PPG' column exists
                if 'PPG' in df.columns:
                    self.ppg_signal = df['PPG'].values
                else:
                    # Try to find a column that might contain PPG data
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        column_name = numeric_cols[0]
                        self.ppg_signal = df[column_name].values
                        QMessageBox.warning(
                            self, "Column Selection", 
                            f"'PPG' column not found. Using '{column_name}' instead."
                        )
                    else:
                        raise ValueError("No numeric columns found in the CSV file")
                
                # Check for NaN values
                if np.isnan(self.ppg_signal).any():
                    # Replace NaN with interpolated values
                    self.ppg_signal = pd.Series(self.ppg_signal).interpolate().values
                    QMessageBox.warning(
                        self, "Data Warning", 
                        "NaN values found in the data and have been interpolated."
                    )
                
                # Update UI
                self.current_filename = os.path.basename(file_path)
                self.status_label.setText(
                    f"Loaded: {self.current_filename} ({len(self.ppg_signal)} samples)"
                )
                
                # Plot the signal
                self.ppg_canvas.plot_signal(self.ppg_signal)
                
                # Enable analyze button
                self.analyze_button.setEnabled(True)
                self.clear_button.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "File Loading Error", 
                                    f"Could not load the file: {str(e)}")
                self.status_label.setText(f"Error loading file: {str(e)}")
    
    def analyze_signal(self):
        """Analyze the loaded PPG signal for AF detection"""
        if self.ppg_signal is None:
            QMessageBox.warning(self, "No Data", "Please load a PPG signal first.")
            return
            
        if self.transformer_model is None or self.spectral_model is None:
            QMessageBox.warning(
                self, "Models Not Loaded", 
                "Required models are not loaded. Please check the application setup."
            )
            return
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Analyzing signal...")
        
        # Disable buttons during analysis
        self.analyze_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        # Create and start the worker thread
        self.worker = AnalysisWorker(
            self.ppg_signal, 
            self.transformer_model, 
            self.spectral_model,
            self.scaler,
            self.selected_indices,
            self.w_transformer
        )
        
        # Connect signals
        self.worker.progress_update.connect(self.update_progress)
        self.worker.analysis_complete.connect(self.display_results)
        self.worker.error_occurred.connect(self.handle_analysis_error)
        
        # Start the worker
        self.worker.start()
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    def display_results(self, segment_results, segment_indices, segment_size):
        """Display the analysis results"""
        # Re-enable buttons
        self.analyze_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Highlight AF segments on the plot
        self.ppg_canvas.highlight_af_segments(segment_results, segment_indices, segment_size)
        
        # Calculate summary statistics
        total_segments = len(segment_results)
        af_segments = sum(1 for prob in segment_results if prob > 0.5)
        af_percentage = (af_segments / total_segments) * 100 if total_segments > 0 else 0
        
        # Update summary label
        summary_text = (
            f"<b>Analysis Results:</b><br>"
            f"File: {self.current_filename}<br>"
            f"Signal Duration: {len(self.ppg_signal)/125:.1f} seconds<br>"
            f"Total Segments: {total_segments}<br>"
            f"Segments with AF: {af_segments} ({af_percentage:.1f}%)<br>"
            f"Model Weights: Transformer {self.w_transformer:.1f} / "
            f"Spectral {1-self.w_transformer:.1f}<br>"
        )
        
        if af_segments > 0:
            summary_text += f"<br><b>AF Detected</b> in {af_segments} segments"
            max_prob = max(segment_results)
            summary_text += f"<br>Max AF Probability: {max_prob:.2f}"
        else:
            summary_text += "<br><b>No AF Detected</b>"
        
        self.summary_label.setText(summary_text)
        
        # Update segments label with detailed information
        segments_text = "<b>Segment Details:</b><br><br>"
        
        for i, (prob, index) in enumerate(zip(segment_results, segment_indices)):
            # Calculate time information
            start_time = index / self.ppg_canvas.fs
            end_time = start_time + (segment_size / self.ppg_canvas.fs)
            
            # Format the segment information
            if prob > 0.5:  # AF segment
                segments_text += f"<span style='color:red'><b>Segment {i+1}: AF Detected</b></span><br>"
                segments_text += f"Time: {start_time:.1f}s - {end_time:.1f}s<br>"
                segments_text += f"Probability: {prob:.3f}<br><br>"
            else:  # Normal segment
                segments_text += f"Segment {i+1}: Normal<br>"
                segments_text += f"Time: {start_time:.1f}s - {end_time:.1f}s<br>"
                segments_text += f"Probability: {prob:.3f}<br><br>"
        
        self.segments_label.setText(segments_text)
        
        # Update status
        if af_segments > 0:
            self.status_label.setText(f"Analysis complete: AF detected in {af_segments} segments")
        else:
            self.status_label.setText("Analysis complete: No AF detected")
    
    def handle_analysis_error(self, error_message):
        """Handle errors during analysis"""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable buttons
        self.analyze_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        # Show error message
        QMessageBox.critical(self, "Analysis Error", 
                            f"An error occurred during analysis: {error_message}")
        self.status_label.setText(f"Error: {error_message}")
    
    def export_results(self):
        """Export analysis results to CSV file"""
        if not hasattr(self.ppg_canvas, 'segment_results') or self.ppg_canvas.segment_results is None:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
        
        # Ask for save location
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self, "Save Results", "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if save_path:
            try:
                # Create results dataframe
                results_data = []
                fs = self.ppg_canvas.fs
                
                for i, (prob, index) in enumerate(zip(self.ppg_canvas.segment_results, 
                                                     self.ppg_canvas.segment_indices)):
                    segment_size = getattr(self.worker, 'segment_size', 1250)
                    
                    start_time = index / fs
                    end_time = start_time + (segment_size / fs)
                    is_af = "Yes" if prob > 0.5 else "No"
                    
                    results_data.append({
                        'Segment': i+1,
                        'Start_Time_s': start_time,
                        'End_Time_s': end_time,
                        'AF_Probability': prob,
                        'Is_AF': is_af
                    })
                
                # Create and save dataframe
                results_df = pd.DataFrame(results_data)
                results_df.to_csv(save_path, index=False)
                
                self.status_label.setText(f"Results exported to {os.path.basename(save_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                                    f"Could not export results: {str(e)}")
                self.status_label.setText(f"Error exporting results: {str(e)}")
    
    def clear_display(self):
        """Clear the current display and data"""
        # Clear plot
        self.ppg_canvas.clear()
        
        # Clear data
        self.ppg_signal = None
        self.current_filename = None
        
        # Reset UI elements
        self.summary_label.setText("No results available")
        self.segments_label.setText("Load a PPG file and analyze to see results")
        self.status_label.setText("Ready")
        
        # Disable buttons
        self.analyze_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    main_window = AFDetectionGUI()
    main_window.show()
    
    # Start application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()