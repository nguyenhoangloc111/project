import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def create_advanced_features(df):
    """
    Create advanced behavioral and temporal features for anomaly detection
    """
    df_features = df.copy()
    
    # === TEMPORAL FEATURES ===
    if 'timestamp' in df_features.columns:
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] < 6)).astype(int)
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_working_hour'] = ((df_features['hour'] >= 9) & (df_features['hour'] < 17)).astype(int)
    else:
        df_features['hour'] = df_features.get('hour', 0)
        df_features['is_night'] = df_features.get('is_night', 0)
        df_features['day_of_week'] = 0
        df_features['is_weekend'] = 0
        df_features['is_working_hour'] = 0
    
    # === IP & LOCATION FEATURES ===
    if 'IP Address' in df_features.columns:
        df_features['ip_hash'] = df_features['IP Address'].apply(
            lambda x: (abs(hash(str(x))) % 100) / 100.0 if pd.notna(x) else 0
        )
    else:
        df_features['ip_hash'] = 0
    
    # === BEHAVIORAL FEATURES ===
    if 'login_frequency' in df_features.columns:
        max_freq = df_features['login_frequency'].max()
        if max_freq > 0:
            df_features['login_freq_norm'] = df_features['login_frequency'] / (max_freq + 1)
        else:
            df_features['login_freq_norm'] = df_features['login_frequency'] * 0
    else:
        df_features['login_frequency'] = 0
        df_features['login_freq_norm'] = 0
    
    # Night + High frequency = suspicious
    if 'is_night' in df_features.columns and 'login_freq_norm' in df_features.columns:
        df_features['night_high_freq'] = df_features['is_night'] * df_features['login_freq_norm']
    else:
        df_features['night_high_freq'] = 0
    
    # === TIME DELTA FEATURES ===
    if 'time_delta' in df_features.columns:
        max_delta = df_features['time_delta'].max()
        if max_delta > 0:
            df_features['time_delta_norm'] = df_features['time_delta'] / (max_delta + 1)
        else:
            df_features['time_delta_norm'] = df_features['time_delta'] * 0
        df_features['quick_login'] = (df_features['time_delta'] < 60).astype(int)
        df_features['long_gap'] = (df_features['time_delta'] > 86400).astype(int)
    else:
        df_features['time_delta'] = 0
        df_features['time_delta_norm'] = 0
        df_features['quick_login'] = 0
        df_features['long_gap'] = 0
    
    # === LOCATION & DEVICE CHANGES ===
    df_features['location_change'] = df_features.get('location_change', 0)
    df_features['device_change'] = df_features.get('device_change', 0)
    
    # Multiple changes = suspicious
    df_features['multiple_changes'] = (
        (df_features['location_change'].fillna(0) + df_features['device_change'].fillna(0)) > 1
    ).astype(int)
    
    # === LOGIN RESULT ===
    df_features['login_result'] = df_features.get('login_result', 1)
    
    # Failed login + changes = high risk
    df_features['failed_with_changes'] = (
        (df_features['login_result'] == 0) * df_features['multiple_changes']
    ).astype(int)
    
    return df_features


def preprocess_login_data(df):
    """
    Preprocess login data with advanced feature engineering
    """
    # Create advanced features
    df_features = create_advanced_features(df)
    
    # Define feature columns
    feature_columns = [
        'hour', 'is_night', 'is_weekend', 'is_working_hour',
        'login_freq_norm', 'night_high_freq',
        'location_change', 'device_change',
        'quick_login', 'long_gap', 'time_delta_norm',
        'multiple_changes', 'failed_with_changes',
        'login_result', 'ip_hash'
    ]
    
    # Select only available columns
    available_cols = [col for col in feature_columns if col in df_features.columns]
    df_selected = df_features[available_cols].copy()
    
    # Handle missing values
    df_selected = df_selected.fillna(0)
    
    # Convert to numeric
    for col in df_selected.columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0)
    
    # Normalize using StandardScaler
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_selected)
        df_normalized = pd.DataFrame(
            features_scaled,
            columns=df_selected.columns,
            index=df_selected.index
        )
        return df_normalized
    except Exception as e:
        print(f"Warning: Normalization failed: {e}")
        return df_selected


def load_and_preprocess_csv(filepath):
    """Read CSV file and preprocess"""
    df = pd.read_csv(filepath)
    return preprocess_login_data(df)


def load_csv_with_labels(filepath):
    """Read CSV file with labels"""
    df = pd.read_csv(filepath)
    y_true = df['is_abnormal'].values if 'is_abnormal' in df.columns else None
    X_preprocessed = preprocess_login_data(df)
    return X_preprocessed, y_true
