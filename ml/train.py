import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from preprocess import load_and_preprocess_csv

def train_isolation_forest(X_train, contamination=0.05):
    """
    Train Isolation Forest model
    
    Args:
        X_train: DataFrame or array containing training data
        contamination: expected proportion of outliers in dataset (0.0 - 1.0)
        Default: 0.02 (only 2% of data expected to be anomalies)
        
    Returns:
        Trained Isolation Forest model
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_train)
    return model


def save_model(model, filepath):
    """
    Save model to file
    
    Args:
        model: model to save
        filepath: file path to save to
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from file
    
    Args:
        filepath: model file path
        
    Returns:
        Loaded Isolation Forest model
    """
    model = joblib.load(filepath)
    return model


if __name__ == "__main__":
    # Example usage
    # df = load_and_preprocess_csv('data/login_logs.csv')
    # model = train_isolation_forest(df)
    # save_model(model, 'model/isolation_forest.joblib')
    pass
