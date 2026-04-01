import pandas as pd
import numpy as np
try:
    from preprocess import preprocess_login_data
except ImportError:
    from .preprocess import preprocess_login_data

def predict_anomaly(model, X_test):
    """
    Predict anomalous behavior with heuristic rules
    
    Args:
        model: trained Isolation Forest model
        X_test: DataFrame or array containing data to predict
        
    Returns:
        Array containing prediction results (-1: anomaly, 1: normal)
    """
    ml_predictions = model.predict(X_test)
    anomaly_scores = model.score_samples(X_test)
    
    # Apply heuristic-based refinement to reduce false positives
    heuristic_predictions = ml_predictions.copy()
    
    # Convert to DataFrame if needed
    if isinstance(X_test, np.ndarray):
        X_df = pd.DataFrame(X_test, columns=['hour', 'is_night', 'login_frequency', 
                                              'location_change', 'device_change', 
                                              'login_result', 'time_delta'])
    else:
        X_df = X_test
    
    for idx in range(len(X_df)):
        row = X_df.iloc[idx]
        
        # Count suspicious factors
        suspicious_count = 0
        if row['login_result'] == 1:  # Failed login
            suspicious_count += 2
        if row['location_change'] == 1:  # Location change
            suspicious_count += 1
        if row['device_change'] == 1:  # Device change
            suspicious_count += 1
        if row['is_night'] == 1:  # Night login
            suspicious_count += 1
        if row['login_frequency'] >= 10:  # High frequency
            suspicious_count += 1
        
        # Apply heuristic logic to override ML prediction
        if suspicious_count >= 2:
            # Strong suspicious indicators = anomaly
            heuristic_predictions[idx] = -1
        elif suspicious_count == 0:
            # No suspicious factors = normal (override any ML detection)
            heuristic_predictions[idx] = 1
        else:
            # Borderline (1 suspicious): use refined ML threshold
            ANOMALY_THRESHOLD = -0.65
            if anomaly_scores[idx] < ANOMALY_THRESHOLD:
                heuristic_predictions[idx] = -1
            else:
                heuristic_predictions[idx] = 1
    
    return heuristic_predictions


def get_anomaly_scores(model, X_test):
    """
    Get anomaly scores
    
    Args:
        model: trained Isolation Forest model
        X_test: DataFrame or array containing data to predict
        
    Returns:
        Array containing anomaly scores
    """
    scores = model.score_samples(X_test)
    return scores




def predict_single_login(model, login_data):
    """
    Predict a single login using ML model with heuristic rules
    
    Args:
        model: trained Isolation Forest model
        login_data: dict or list containing [hour, is_night, login_frequency, 
                    location_change, device_change, login_result, time_delta]
        
    Returns:
        dict containing prediction, anomaly_score, confidence, is_anomaly
    """
    if isinstance(login_data, dict):
        data = pd.DataFrame([login_data])
        features = login_data
    else:
        columns = ['hour', 'is_night', 'login_frequency', 'location_change', 
                   'device_change', 'login_result', 'time_delta']
        data = pd.DataFrame([login_data], columns=columns)
        features = {col: login_data[i] for i, col in enumerate(columns)}
    
    data = preprocess_login_data(data)
    
    prediction = model.predict(data)[0]
    anomaly_score = model.score_samples(data)[0]
    
    # Apply business logic heuristics
    # Count suspicious factors
    suspicious_count = 0
    if features.get('login_result', 0) == 1:  # Failed login
        suspicious_count += 2  # High weight
    if features.get('location_change', 0) == 1:  # Location change
        suspicious_count += 1
    if features.get('device_change', 0) == 1:  # Device change
        suspicious_count += 1
    if features.get('is_night', 0) == 1:  # Night login
        suspicious_count += 1
    if features.get('login_frequency', 0) >= 10:  # High frequency
        suspicious_count += 1
    
    # Determine anomaly based on suspicious factors (primary) + ML score (secondary)
    # Primary: if 2+ suspicious factors, likely anomaly
    # Secondary: use ML score if primary unclear
    if suspicious_count >= 2:
        is_anomaly = True
        confidence = min(100, suspicious_count * 25)
    elif suspicious_count == 1:
        # Borderline case - use ML score as tiebreaker
        ANOMALY_THRESHOLD = -0.65  # Refined threshold
        is_anomaly = anomaly_score < ANOMALY_THRESHOLD
        confidence = min(100, abs(anomaly_score - ANOMALY_THRESHOLD) * 500)
    else:
        # No suspicious factors - likely normal
        is_anomaly = False
        confidence = 85  # High confidence in normal
    
    return {
        'prediction': int(prediction),  # 1: normal, -1: anomaly (from model)
        'anomaly_score': float(anomaly_score),
        'is_anomaly': is_anomaly,  # Based on heuristics + ML
        'confidence': float(confidence)
    }
