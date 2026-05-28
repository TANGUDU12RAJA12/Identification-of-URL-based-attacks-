"""
Data preprocessing utilities for the cyber attack detection app
"""

import pandas as pd
import numpy as np
import io

def preprocess_data(df, feature_names, scaler):
    """
    Preprocess the input data for prediction
    """
    # Check if required features exist
    available_features = [f for f in feature_names if f in df.columns]
    
    # Extract features
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle negative values
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].clip(lower=0)
    
    # Log transform for skewed features
    skewed_cols = ['source_ip_freq', 'destination_ip_freq', 'payload_special_chars', 
                   'response_size', 'url_length']
    for col in skewed_cols:
        if col in X.columns:
            X[col] = X[col].clip(lower=0)
            X[col] = np.log1p(X[col])
    
    # Handle outliers
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            X[col] = X[col].clip(q01, q99)
    
    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, X

def parse_uploaded_file(uploaded_file):
    """
    Parse uploaded CSV or TXT file
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            # Try to read as CSV first
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
        else:
            return None, "Unsupported file format. Please upload CSV or TXT file."
        
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"