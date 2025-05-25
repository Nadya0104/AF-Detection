"""
AF Detection Dashboard using Plotly Dash
Replacement for PyQt5 GUI with better visualization capabilities
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
from models.transformer import load_transformer_model
from models.spectral import load_spectral_model 
from models.transformer import predict_transformer
from models.spectral import  predict_spectral

# Import your existing models (adjust paths based on your project structure)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Global variables for models
transformer_model = None
spectral_model = None
scaler = None
selected_indices = None

# Load models at startup
def load_models():
    global transformer_model, spectral_model, scaler, selected_indices
    try:
        transformer_model = load_transformer_model("saved_transformer_model/transformer_model.pth", 'cpu')
        spectral_model, scaler, selected_indices = load_spectral_model(
            "saved_spectral_models/best_model.pkl",
            "saved_spectral_models/scaler.pkl", 
            "saved_spectral_models/selected_indices.pkl"
        )
        return "Models loaded successfully"
    except Exception as e:
        return f"Error loading models: {str(e)}"

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("AF Detection from PPG Signals", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        html.Hr(),
    ]),
    
    # Control Panel
    html.Div([
        html.Div([
            # File upload
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select PPG CSV File')
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
        ], className='four columns'),
        
        html.Div([
            # Model weight selection
            html.Label("Model Weighting:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='weight-dropdown',
                options=[
                    {'label': '50/50 Blend', 'value': 0.5},
                    {'label': 'Favor Transformer (70%)', 'value': 0.7},
                    {'label': 'Favor Spectral (30%)', 'value': 0.3},
                    {'label': 'Transformer Only', 'value': 1.0},
                    {'label': 'Spectral Only', 'value': 0.0}
                ],
                value=0.5,
                style={'marginBottom': 10}
            ),
        ], className='four columns'),
        
        html.Div([
            # Analysis button
            html.Button('Analyze Signal', id='analyze-btn', 
                       style={'width': '100%', 'height': '60px', 'fontSize': '16px',
                             'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                             'borderRadius': '5px', 'marginTop': '25px'},
                       disabled=True),
        ], className='four columns'),
    ], className='row', style={'marginBottom': 20}),
    
    # Progress bar
    html.Div([
        dcc.Interval(id='progress-interval', interval=500, disabled=True),
        html.Div(id='progress-bar', style={'display': 'none'})
    ]),
    
    # Status message
    html.Div(id='status-message', 
             style={'textAlign': 'center', 'marginBottom': 20, 'fontSize': '14px'}),
    
    # Main visualization area
    html.Div([
        # PPG Signal Plot
        html.Div([
            dcc.Graph(id='ppg-signal-plot', style={'height': '400px'})
        ], className='twelve columns'),
    ], className='row'),
    
    # Results section
    html.Div([
        # Summary statistics
        html.Div([
            html.H4("Analysis Summary", style={'color': '#2c3e50'}),
            html.Div(id='summary-stats')
        ], className='six columns'),
        
        # Detailed results
        html.Div([
            html.H4("Segment Details", style={'color': '#2c3e50'}),
            html.Div(id='segment-details', style={'maxHeight': '300px', 'overflowY': 'auto'})
        ], className='six columns'),
    ], className='row', style={'marginTop': 20}),
    
    # Additional visualizations
    html.Div([
        # Probability distribution
        html.Div([
            dcc.Graph(id='probability-dist', style={'height': '300px'})
        ], className='six columns'),
        
        # Feature importance (if available)
        html.Div([
            dcc.Graph(id='feature-importance', style={'height': '300px'})
        ], className='six columns'),
    ], className='row', style={'marginTop': 20}),
    
    # Hidden div to store data
    html.Div(id='stored-data', style={'display': 'none'}),
    html.Div(id='analysis-results', style={'display': 'none'})
])

# Callback for file upload
@app.callback(
    [Output('stored-data', 'children'),
     Output('analyze-btn', 'disabled'),
     Output('status-message', 'children'),
     Output('ppg-signal-plot', 'figure')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_data(contents, filename):
    if contents is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Load PPG data to begin analysis")
        return None, True, load_models(), empty_fig
    
    try:
        # Parse uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Read CSV
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Extract PPG signal
        if 'PPG' in df.columns:
            ppg_signal = df['PPG'].values
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ppg_signal = df[numeric_cols[0]].values
            else:
                raise ValueError("No numeric columns found")
        
        # Handle NaN values
        if np.isnan(ppg_signal).any():
            ppg_signal = pd.Series(ppg_signal).interpolate().values
        
        # Create time axis
        fs = 125  # Sampling frequency
        time_axis = np.arange(len(ppg_signal)) / fs
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=ppg_signal, 
                                mode='lines', name='PPG Signal',
                                line=dict(color='blue', width=1)))
        fig.update_layout(
            title=f"PPG Signal: {filename}",
            xaxis_title="Time (seconds)",
            yaxis_title="PPG Amplitude",
            hovermode='x unified'
        )
        
        # Store data
        data_dict = {
            'ppg_signal': ppg_signal.tolist(),
            'filename': filename,
            'fs': fs,
            'duration': len(ppg_signal) / fs
        }
        
        status = f"Loaded: {filename} ({len(ppg_signal)} samples, {len(ppg_signal)/fs:.1f}s)"
        
        return str(data_dict), False, status, fig
        
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Error loading file")
        return None, True, f"Error loading file: {str(e)}", empty_fig

# Callback for analysis
@app.callback(
    [Output('analysis-results', 'children'),
     Output('status-message', 'children', allow_duplicate=True),
     Output('ppg-signal-plot', 'figure', allow_duplicate=True)],
    [Input('analyze-btn', 'n_clicks')],
    [State('stored-data', 'children'),
     State('weight-dropdown', 'value')],
    prevent_initial_call=True
)
def analyze_signal(n_clicks, stored_data, w_transformer):
    if n_clicks is None or stored_data is None:
        return None, "No analysis performed", go.Figure()
    
    try:
        # Parse stored data
        data_dict = eval(stored_data)
        ppg_signal = np.array(data_dict['ppg_signal'])
        filename = data_dict['filename']
        fs = data_dict['fs']
        
        # Perform analysis (simplified version)
        segment_size = 1250  # 10 seconds at 125Hz
        stride = 625  # 5 seconds overlap
        
        segments = []
        segment_indices = []
        
        for i in range(0, len(ppg_signal) - segment_size + 1, stride):
            segments.append(ppg_signal[i:i + segment_size])
            segment_indices.append(i)
        
        if len(segments) == 0:
            if len(ppg_signal) >= segment_size // 2:
                segments.append(ppg_signal)
                segment_indices.append(0)
            else:
                return None, "Signal too short for analysis", go.Figure()
        
        # Analyze each segment
        segment_results = []
        for segment in segments:
            # Get transformer prediction
            transformer_prob = predict_transformer(segment, transformer_model)
            
            # Get spectral prediction  
            spectral_prob = predict_spectral(segment, spectral_model, scaler, selected_indices)
            
            # Weighted fusion
            final_prob = (w_transformer * transformer_prob) + ((1 - w_transformer) * spectral_prob)
            segment_results.append(final_prob)
        
        # Create updated plot with highlighted segments
        time_axis = np.arange(len(ppg_signal)) / fs
        fig = go.Figure()
        
        # Plot signal
        fig.add_trace(go.Scatter(x=time_axis, y=ppg_signal, 
                                mode='lines', name='PPG Signal',
                                line=dict(color='blue', width=1)))
        
        # Highlight AF segments
        for i, (prob, index) in enumerate(zip(segment_results, segment_indices)):
            start_time = index / fs
            end_time = start_time + (segment_size / fs)
            
            if prob > 0.5:  # AF detected
                fig.add_vrect(
                    x0=start_time, x1=end_time,
                    fillcolor="red", opacity=0.3,
                    layer="below", line_width=0,
                )
                
                # Add annotation
                fig.add_annotation(
                    x=start_time + (end_time - start_time) / 2,
                    y=max(ppg_signal) * 0.9,
                    text=f"AF ({prob*100:.0f}%)",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="red"
                )
        
        fig.update_layout(
            title=f"PPG Signal with AF Detection: {filename}",
            xaxis_title="Time (seconds)",
            yaxis_title="PPG Amplitude",
            hovermode='x unified'
        )
        
        # Store results
        results_dict = {
            'segment_results': segment_results,
            'segment_indices': segment_indices,
            'segment_size': segment_size,
            'total_segments': len(segment_results),
            'af_segments': sum(1 for prob in segment_results if prob > 0.5),
            'filename': filename
        }
        
        af_count = results_dict['af_segments']
        status = f"Analysis complete: {'AF detected' if af_count > 0 else 'No AF detected'} ({af_count} segments)"
        
        return str(results_dict), status, fig
        
    except Exception as e:
        return None, f"Analysis error: {str(e)}", go.Figure()

# Callback for updating summary stats
@app.callback(
    [Output('summary-stats', 'children'),
     Output('segment-details', 'children'),
     Output('probability-dist', 'figure')],
    [Input('analysis-results', 'children')]
)
def update_results_display(analysis_results):
    if analysis_results is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No analysis results")
        return "No results available", "Load and analyze PPG data to see results", empty_fig
    
    try:
        results = eval(analysis_results)
        
        # Summary statistics
        total_segments = results['total_segments']
        af_segments = results['af_segments']
        af_percentage = (af_segments / total_segments) * 100 if total_segments > 0 else 0
        
        summary = html.Div([
            html.P(f"File: {results['filename']}", style={'fontWeight': 'bold'}),
            html.P(f"Total Segments: {total_segments}"),
            html.P(f"AF Segments: {af_segments} ({af_percentage:.1f}%)"),
            html.P(f"Status: {'AF Detected' if af_segments > 0 else 'Normal'}", 
                   style={'color': 'red' if af_segments > 0 else 'green', 'fontWeight': 'bold'})
        ])
        
        # Detailed segment information
        segment_details = []
        for i, (prob, index) in enumerate(zip(results['segment_results'], results['segment_indices'])):
            start_time = index / 125  # Assuming 125Hz
            end_time = start_time + (results['segment_size'] / 125)
            
            color = 'red' if prob > 0.5 else 'green'
            status = 'AF' if prob > 0.5 else 'Normal'
            
            segment_details.append(
                html.P(f"Segment {i+1}: {status} ({prob:.3f}) - {start_time:.1f}s to {end_time:.1f}s",
                       style={'color': color, 'margin': '5px 0'})
            )
        
        # Probability distribution plot
        probs = results['segment_results']
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=probs, nbinsx=20, name='Probability Distribution'))
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                      annotation_text="AF Threshold")
        fig.update_layout(
            title="AF Probability Distribution",
            xaxis_title="Probability",
            yaxis_title="Count"
        )
        
        return summary, segment_details, fig
        
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Error displaying results")
        return f"Error: {str(e)}", "Error displaying results", empty_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)