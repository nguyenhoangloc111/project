import os
import sys
import argparse

# Add ml module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from ml.train import train_isolation_forest, save_model
from ml.preprocess import load_and_preprocess_csv

def main(contamination=0.02, data_path='data/rfff-dataset.csv', model_path='model/isolation_forest.joblib'):
    """
    Script to train Isolation Forest model from CSV data
    
    Args:
        contamination (float): Contamination rate (0.01-0.03) - expected proportion of anomalies
        data_path (str): Path to training data CSV file
        model_path (str): Path where trained model will be saved
    """
    # ✅ Validate contamination range (0.01-0.03)
    if contamination < 0.01 or contamination > 0.03:
        print(f"❌ Error: Contamination must be between 0.01 and 0.03")
        print(f"   Received: {contamination}")
        return False
    
    print("=" * 60)
    print("Isolation Forest - Login Anomaly Detection Model Training")
    print("=" * 60)
    
    # Check data file
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file '{data_path}' not found!")
        return False
    
    print(f"\n📂 Loading data from: {data_path}")
    
    # Data preprocessing
    df = load_and_preprocess_csv(data_path)
    
    # Check loaded data
    if df is None or df.empty:
        print(f"❌ Error: Data is empty or invalid!")
        return False
    print(f"✓ Loaded {len(df):,} records")
    print(f"✓ Features: {list(df.columns)}")
    
    # Train model with specified contamination rate
    print(f"\n🚀 Training Isolation Forest model...")
    print(f"   Contamination rate: {contamination} ({contamination*100:.1f}%)")
    print(f"   Expected anomalies: ~{int(len(df) * contamination):,}")
    
    model = train_isolation_forest(df, contamination=contamination)
    print("✓ Model trained successfully!")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    save_model(model, model_path)
    
    print(f"\n✓ Model saved at: {model_path}")
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Isolation Forest model for login anomaly detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                    # Use default contamination (0.02)
  python train_model.py --contamination 0.01  # High security
  python train_model.py --contamination 0.03  # Sensitive detection
        """
    )
    
    parser.add_argument(
        '--contamination', 
        type=float, 
        default=0.02,
        help='Contamination rate: expected proportion of anomalies (0.01-0.03, default: 0.02)'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/rfff-dataset.csv',
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='model/isolation_forest.joblib',
        help='Path where trained model will be saved'
    )
    
    args = parser.parse_args()
    
    # Run training
    success = main(
        contamination=args.contamination,
        data_path=args.data,
        model_path=args.output
    )
    
    sys.exit(0 if success else 1)
