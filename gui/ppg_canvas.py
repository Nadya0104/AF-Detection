"""
Canvas widget for plotting PPG signals and AF detection results
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PPGCanvas(FigureCanvas):
    """Canvas for displaying PPG signal with AF detection results"""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super(PPGCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Initialize with empty data
        self.signal_data = None
        self.time_data = None
        self.segment_results = None
        self.segment_indices = None
        self.fs = 125  # Default sampling frequency
        
        # Setup basic plot
        self.signal_line = None
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the initial empty plot"""
        self.axes.clear()
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('PPG Amplitude')
        self.axes.set_title('PPG Signal with AF Detection')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        self.fig.canvas.draw()
        
    def plot_signal(self, signal_data, fs=125):
        """Plot the PPG signal"""
        self.signal_data = signal_data
        self.fs = fs
        self.time_data = np.arange(len(signal_data)) / fs
        
        self.axes.clear()
        self.signal_line, = self.axes.plot(self.time_data, self.signal_data, 'b-', linewidth=1)
        
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('PPG Amplitude')
        self.axes.set_title('PPG Signal with AF Detection')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Set limits with some padding
        y_range = np.max(signal_data) - np.min(signal_data)
        self.axes.set_ylim(np.min(signal_data) - 0.1*y_range, np.max(signal_data) + 0.1*y_range)
        
        self.fig.canvas.draw()
        
    def highlight_af_segments(self, segment_results, segment_indices, segment_size):
        """Highlight segments with AF"""
        self.segment_results = segment_results
        self.segment_indices = segment_indices
        
        # Ensure we have a plotted signal
        if self.signal_data is None:
            return
            
        # Loop through segments and highlight those with AF
        for i, (prob, index) in enumerate(zip(segment_results, segment_indices)):
            # Convert segment index to time
            start_time = index / self.fs
            # Calculate segment duration in seconds
            segment_duration = segment_size / self.fs
            
            # Determine color based on probability (red = AF, green = normal)
            if prob > 0.5:  # AF detected
                # Draw a semi-transparent red rectangle
                rect = self.axes.axvspan(start_time, start_time + segment_duration, 
                                         alpha=0.3, color='red', label=f'AF ({prob:.2f})')
                
                # Add text annotation
                confidence = prob * 100
                self.axes.text(start_time + segment_duration/2, self.axes.get_ylim()[1] * 0.95, 
                              f'AF ({confidence:.0f}%)', color='red', fontweight='bold',
                              ha='center', backgroundcolor='white', alpha=0.7)
        
        # Update the plot
        self.fig.canvas.draw()
    
    def clear(self):
        """Clear the plot"""
        self.axes.clear()
        self.setup_plot()
        self.signal_data = None
        self.time_data = None
        self.segment_results = None
        self.segment_indices = None