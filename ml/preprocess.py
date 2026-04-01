import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_login_data(df):
    """
    Preprocess login data
    
    Args:
        df: DataFrame containing login data
        
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    # Process hour and is_night columns if not already present
    if 'timestamp' in df_processed.columns and 'hour' not in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['is_night'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] < 6)).astype(int)
    
    # Ensure required columns exist
    required_columns = ['hour', 'is_night', 'login_frequency', 'location_change', 
                       'device_change', 'login_result', 'time_delta']
    
    for col in required_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Handle missing values
    df_processed = df_processed.fillna(0)
    
    # Convert data types
    for col in required_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    return df_processed[required_columns]


def load_and_preprocess_csv(filepath):
    """
    Read CSV file and preprocess
    
    Args:
        filepath: path to CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)
    return preprocess_login_data(df)
