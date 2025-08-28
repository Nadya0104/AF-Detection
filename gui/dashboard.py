"""
AF Detection Dashboard using Plotly Dash
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import sys
import os
from models.transformer import predict_transformer
from models.spectral import  predict_spectral

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Global variables for models
TRANSFORMER_MODEL_DIR = "saved_transformer_model"
SPECTRAL_MODEL_DIR = "saved_spectral_model"

# Check if models are available at startup
def check_models():
    try:
        # Check if required model files exist
        transformer_files = [
            os.path.join(TRANSFORMER_MODEL_DIR, 'transformer_model.pth'),
            os.path.join(TRANSFORMER_MODEL_DIR, 'model_config.pth')
        ]
        
        spectral_files = [
            os.path.join(SPECTRAL_MODEL_DIR, 'best_model.pkl'),
            os.path.join(SPECTRAL_MODEL_DIR, 'scaler.pkl'),
            os.path.join(SPECTRAL_MODEL_DIR, 'selected_indices.pkl')
        ]
        
        missing_files = []
        for file in transformer_files + spectral_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            return f"Missing model files: {', '.join(missing_files)}"
        else:
            return "Models available - ready for analysis"
            
    except Exception as e:
        return f"Error checking models: {str(e)}"

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
                    {'label': 'Favor Transformer (70/30)', 'value': 0.7},
                    {'label': 'Favor Spectral (70/30)', 'value': 0.3},
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
        return None, True, check_models(), empty_fig
    
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
        
        # Perform analysis
        # Get predictions using the proper inference functions
        try:
            # Get transformer prediction 
            transformer_prob = predict_transformer(ppg_signal, TRANSFORMER_MODEL_DIR)
            print(f"Transformer prediction: {transformer_prob:.3f}")
            
            # Get spectral prediction 
            spectral_prob = predict_spectral(ppg_signal, SPECTRAL_MODEL_DIR)
            print(f"Spectral prediction: {spectral_prob:.3f}")
            
            # Weighted fusion
            final_prob = (w_transformer * transformer_prob) + ((1 - w_transformer) * spectral_prob)
            print(f"Final weighted prediction: {final_prob:.3f}")
            
            segment_size = 1250  # 10 seconds at 125Hz
            stride = 625  
            
            display_segments = []
            segment_indices = []
            
            for i in range(0, len(ppg_signal) - segment_size + 1, stride):
                display_segments.append(ppg_signal[i:i + segment_size])
                segment_indices.append(i)
            
            if len(display_segments) == 0:
                if len(ppg_signal) >= segment_size // 2:
                    display_segments.append(ppg_signal)
                    segment_indices.append(0)
            
            # For visualization, assign the final prediction to all segments
            segment_results = [final_prob] * len(display_segments)
            
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            return None, f"Prediction error: {str(pred_error)}", go.Figure()
        
        # Create updated plot with highlighted segments
        time_axis = np.arange(len(ppg_signal)) / fs
        fig = go.Figure()
        
        # Plot signal
        fig.add_trace(go.Scatter(x=time_axis, y=ppg_signal, 
                                mode='lines', name='PPG Signal',
                                line=dict(color='blue', width=1)))
        
        # Highlight AF segments based on overall prediction
        if final_prob > 0.5:  # AF detected
            # Highlight the entire signal or segments
            for i, (prob, index) in enumerate(zip(segment_results, segment_indices)):
                start_time = index / fs
                end_time = start_time + (segment_size / fs)
                
                fig.add_vrect(
                    x0=start_time, x1=end_time,
                    fillcolor="red", opacity=0.3,
                    layer="below", line_width=0,
                )
                
                # Add annotation only for first segment to avoid clutter
                if i == 0:
                    fig.add_annotation(
                        x=start_time + (end_time - start_time) / 2,
                        y=max(ppg_signal) * 0.9,
                        text=f"AF Detected ({final_prob*100:.1f}%)",
                        showarrow=False,
                        bgcolor="white",
                        bordercolor="red"
                    )
        else:
            # Add annotation for normal
            if len(segment_indices) > 0:
                mid_time = len(ppg_signal) / fs / 2
                fig.add_annotation(
                    x=mid_time,
                    y=max(ppg_signal) * 0.9,
                    text=f"Normal ({final_prob*100:.1f}%)",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="green"
                )
        
        fig.update_layout(
            title=f"PPG Signal with AF Detection: {filename}",
            xaxis_title="Time (seconds)",
            yaxis_title="PPG Amplitude",
            hovermode='x unified'
        )
        
        # Store results
        results_dict = {
            'transformer_prob': transformer_prob,
            'spectral_prob': spectral_prob,
            'final_prob': final_prob,
            'segment_results': segment_results,  
            'segment_indices': segment_indices,  
            'segment_size': segment_size,
            'total_segments': len(segment_results),
            'af_detected': final_prob > 0.5,
            'filename': filename
        }

        status = f"Analysis complete: {'AF detected' if final_prob > 0.5 else 'No AF detected'} (confidence: {final_prob*100:.1f}%)"
        
        return str(results_dict), status, fig
        
    except Exception as e:
        return None, f"Analysis error: {str(e)}", go.Figure()


@app.callback(
    Output('summary-stats', 'children'),  
    [Input('analysis-results', 'children')]
)
def update_results_display(analysis_results):
    if analysis_results is None:
        return "No results available"
    
    try:
        results = eval(analysis_results)
        
        # Summary statistics
        transformer_prob = results['transformer_prob']
        spectral_prob = results['spectral_prob'] 
        final_prob = results['final_prob']
        af_detected = results['af_detected']

        summary = html.Div([
            html.P(f"File: {results['filename']}", style={'fontWeight': 'bold'}),
            html.P(f"Transformer Prediction: {transformer_prob:.3f} ({transformer_prob*100:.1f}%)"),
            html.P(f"Spectral Prediction: {spectral_prob:.3f} ({spectral_prob*100:.1f}%)"),
            html.P(f"Final Prediction: {final_prob:.3f} ({final_prob*100:.1f}%)"),
            html.P(f"Status: {'AF Detected' if af_detected else 'Normal'}", 
                style={'color': 'red' if af_detected else 'green', 'fontWeight': 'bold'})
        ])
        
        return summary
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=False, port=8050)