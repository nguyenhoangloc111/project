from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from datetime import datetime
import sys
from werkzeug.utils import secure_filename
import io
import json
from flask_cors import CORS

# Add ml module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from ml.train import train_isolation_forest, train_isolation_forest_with_history, save_model, load_model, calculate_metrics
from ml.preprocess import load_and_preprocess_csv, preprocess_login_data, load_csv_with_labels
from ml.predict import predict_single_login, predict_anomaly, get_anomaly_scores
from ml.evaluate import calculate_metrics_with_viz

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# ✅ Enable CORS for frontend API calls
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size (increased from 16MB)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model and metrics file paths
MODEL_PATH = 'model/isolation_forest.joblib'
METRICS_PATH = 'model/metrics.json'
MODEL = None

def save_metrics_to_file(metrics):
    """Save metrics to a JSON file for persistence"""
    try:
        os.makedirs('model', exist_ok=True)
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to {METRICS_PATH}")
    except Exception as e:
        print(f"⚠️ Error saving metrics: {e}")

def load_metrics_from_file():
    """Load metrics from JSON file if available"""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            print(f"✅ Metrics loaded from {METRICS_PATH}")
            return metrics
    except Exception as e:
        print(f"⚠️ Error loading metrics: {e}")
    return None

# Function to auto-train model on startup
def auto_train_model():
    """Auto-train model on application startup"""
    global MODEL
    try:
        default_dataset = 'data/rfff-dataset.csv'
        if os.path.exists(default_dataset):
            print("🔄 Auto-training model on startup...")
            df = load_and_preprocess_csv(default_dataset)
            # Use simple training for faster startup (not history version)
            MODEL = train_isolation_forest(df, contamination=0.28)
            os.makedirs('model', exist_ok=True)
            save_model(MODEL, MODEL_PATH)
            print(f"✅ Model trained successfully on startup! ({len(df)} samples)")
            return True
        else:
            print(f"⚠️ Default dataset not found: {default_dataset}")
            return False
    except Exception as e:
        print(f"❌ Error auto-training model: {e}")
        return False

# Load existing model or auto-train if not available
if os.path.exists(MODEL_PATH):
    try:
        MODEL = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting auto-training...")
        auto_train_model()
else:
    print("Model not found. Attempting auto-training...")
    auto_train_model()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train():
    """
    API to train model
    
    Expected JSON:
    {
        "csv_file": "path/to/file.csv",
        "contamination": 0.28
    }
    """
    global MODEL
    
    try:
        data = request.get_json()
        csv_file = data.get('csv_file', 'data/rfff-dataset.csv')
        contamination = 0.28  # Fixed default contamination rate (28% actual anomaly rate)
        
        if not os.path.exists(csv_file):
            return jsonify({'error': f'File not found: {csv_file}'}), 400
        
        # Data preprocessing (with labels)
        X_data, y_true = load_csv_with_labels(csv_file)
        
        # Train model with accuracy history
        MODEL, accuracy_history = train_isolation_forest_with_history(X_data, contamination=contamination)
        
        # Calculate evaluation metrics with visualization data
        metrics = calculate_metrics_with_viz(MODEL, X_data, y_true)
        
        # Save model
        os.makedirs('model', exist_ok=True)
        save_model(MODEL, MODEL_PATH)
        
        # Save metrics to file
        if metrics:
            save_metrics_to_file(metrics)
        
        response = {
            'status': 'success',
            'message': 'Model trained and saved successfully',
            'samples_trained': len(X_data),
            'accuracy_history': accuracy_history
        }
        
        # Add metrics if available
        if metrics:
            response.update(metrics)
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API to predict login behavior
    
    Expected JSON:
    {
        "hour": 2,
        "is_night": 1,
        "login_frequency": 5,
        "location_change": 1,
        "device_change": 1,
        "login_result": 0,
        "time_delta": 30
    }
    """
    global MODEL
    
    if MODEL is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        data = request.get_json()
        print(f"[PREDICT] Received data: {data}")
        
        # Check required fields
        required_fields = ['hour', 'is_night', 'login_frequency', 'location_change', 
                          'device_change', 'login_result', 'time_delta']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prediction
        print(f"[PREDICT] Starting prediction...")
        result = predict_single_login(MODEL, data)
        print(f"[PREDICT] Prediction result: {result}")
        
        return jsonify({
            'status': 'success',
            'prediction': int(result['prediction']),
            'is_anomaly': bool(result['is_anomaly']),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"[PREDICT ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    API to batch predict
    
    Expected JSON:
    {
        "csv_file": "path/to/file.csv"
    }
    """
    global MODEL
    
    if MODEL is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        data = request.get_json()
        csv_file = data.get('csv_file', 'data/rfff-dataset.csv')
        
        if not os.path.exists(csv_file):
            return jsonify({'error': f'File not found: {csv_file}'}), 400
        
        # Load original dataframe to preserve original columns
        df_original = pd.read_csv(csv_file)
        
        # Data preprocessing
        df_processed = load_and_preprocess_csv(csv_file)
        
        # Prediction
        predictions = predict_anomaly(MODEL, df_processed)
        anomaly_scores = get_anomaly_scores(MODEL, df_processed)
        
        # Add results to original dataframe
        df_original['prediction'] = predictions
        df_original['anomaly_score'] = anomaly_scores
        df_original['is_anomaly'] = predictions == -1
        
        # Create statistics
        total_records = len(df_original)
        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()
        
        # Select only columns that exist in the original dataframe for reporting
        report_cols = ['anomaly_score']
        for col in ['is_night', 'login_frequency', 'location_change', 'device_change', 'login_result', 'time_delta']:
            if col in df_original.columns:
                report_cols.append(col)
        
        # Convert anomalies dataframe - convert numpy types to Python types
        anomalies_list = df_original[df_original['is_anomaly']][report_cols].to_dict('records')
        
        # Convert numpy types to Python types
        for record in anomalies_list:
            for key, value in record.items():
                if hasattr(value, 'item'):  # numpy type
                    record[key] = value.item()
        
        return jsonify({
            'status': 'success',
            'total_records': int(total_records),
            'anomaly_count': int(anomaly_count),
            'normal_count': int(normal_count),
            'anomaly_percentage': float(anomaly_count / total_records * 100) if total_records > 0 else 0,
            'anomalies': anomalies_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check system status"""
    model_status = 'trained' if MODEL is not None else 'not_trained'
    
    return jsonify({
        'status': 'ok',
        'model_status': model_status,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model evaluation metrics"""
    metrics = load_metrics_from_file()
    
    if metrics:
        return jsonify({
            'status': 'success',
            'metrics': metrics
        }), 200
    else:
        return jsonify({
            'status': 'not_available',
            'message': 'No metrics available. Please train the model first.'
        }), 200


@app.route('/api/train-upload', methods=['POST'])
def train_upload():
    """
    API to train model from uploaded file
    """
    global MODEL
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File not found'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are accepted'}), 400
        
        contamination = 0.28  # Fixed default contamination rate (28% actual anomaly rate)
        
        # Read file
        try:
            df_raw = pd.read_csv(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Extract labels before preprocessing
        y_true = df_raw['is_abnormal'].values if 'is_abnormal' in df_raw.columns else None
        
        # Data preprocessing
        df = preprocess_login_data(df_raw)
        
        # Train model with accuracy history
        MODEL, accuracy_history = train_isolation_forest_with_history(df, contamination=contamination)
        
        # Calculate evaluation metrics with visualization data
        metrics = calculate_metrics_with_viz(MODEL, df, y_true)
        
        # Save model
        os.makedirs('model', exist_ok=True)
        save_model(MODEL, MODEL_PATH)
        
        # Save metrics to file
        if metrics:
            save_metrics_to_file(metrics)
        
        response = {
            'status': 'success',
            'message': 'Model trained from file successfully',
            'samples_trained': len(df),
            'filename': file.filename,
            'accuracy_history': accuracy_history
        }
        
        # Add metrics if available
        if metrics:
            response.update(metrics)
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-batch-upload', methods=['POST'])
def predict_batch_upload():
    """
    API to batch predict from uploaded file
    """
    global MODEL
    
    if MODEL is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File not found'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are accepted'}), 400
        
        # Read file
        try:
            file_content = file.read()
            df_original = pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Data preprocessing
        from ml.preprocess import preprocess_login_data
        df_processed = preprocess_login_data(df_original.copy())
        
        # Prediction
        predictions = predict_anomaly(MODEL, df_processed)
        anomaly_scores = get_anomaly_scores(MODEL, df_processed)
        
        # Add results to original dataframe
        df_original['prediction'] = predictions
        df_original['anomaly_score'] = anomaly_scores
        df_original['is_anomaly'] = predictions == -1
        
        # Create statistics
        total_records = len(df_original)
        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()
        
        # Select only columns that exist in the original dataframe for reporting
        report_cols = ['anomaly_score']
        for col in ['is_night', 'login_frequency', 'location_change', 'device_change', 'login_result', 'time_delta']:
            if col in df_original.columns:
                report_cols.append(col)
        
        # Convert anomalies dataframe - convert numpy types to Python types
        anomalies_list = df_original[df_original['is_anomaly']][report_cols].to_dict('records')
        
        # Convert numpy types to Python types
        for record in anomalies_list:
            for key, value in record.items():
                if hasattr(value, 'item'):  # numpy type
                    record[key] = value.item()
        
        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'total_records': int(total_records),
            'anomaly_count': int(anomaly_count),
            'normal_count': int(normal_count),
            'anomaly_percentage': float(anomaly_count / total_records * 100) if total_records > 0 else 0,
            'anomalies': anomalies_list
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
