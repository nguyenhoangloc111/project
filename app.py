from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from datetime import datetime
import sys
from werkzeug.utils import secure_filename
import io
from flask_cors import CORS

# Add ml module path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from ml.train import train_isolation_forest, save_model, load_model
from ml.preprocess import load_and_preprocess_csv, preprocess_login_data
from ml.predict import predict_single_login, predict_anomaly, get_anomaly_scores

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# ✅ Enable CORS for frontend API calls
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model file path
MODEL_PATH = 'model/isolation_forest.joblib'
MODEL = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        MODEL = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")


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
        "contamination": 0.02
    }
    """
    global MODEL
    
    try:
        data = request.get_json()
        csv_file = data.get('csv_file', 'data/rfff-dataset.csv')
        contamination = 0.02  # Fixed default contamination rate
        
        if not os.path.exists(csv_file):
            return jsonify({'error': f'File not found: {csv_file}'}), 400
        
        # Data preprocessing
        df = load_and_preprocess_csv(csv_file)
        
        # Train model
        MODEL = train_isolation_forest(df, contamination=contamination)
        
        # Save model
        os.makedirs('model', exist_ok=True)
        save_model(MODEL, MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained and saved successfully',
            'samples_trained': len(df)
        }), 200
        
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
        
        # Data preprocessing
        df = load_and_preprocess_csv(csv_file)
        
        # Prediction
        predictions = predict_anomaly(MODEL, df)
        anomaly_scores = get_anomaly_scores(MODEL, df)
        
        # Add results to dataframe
        df['prediction'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = predictions == -1
        
        # Create statistics
        total_records = len(df)
        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()
        
        # Convert anomalies dataframe - convert numpy types to Python types
        anomalies_list = df[df['is_anomaly']][['is_night', 'login_frequency', 
                                                'location_change', 'device_change', 
                                                'login_result', 'time_delta', 
                                                'anomaly_score']].to_dict('records')
        
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
        
        contamination = 0.02  # Fixed default contamination rate
        
        # Read file
        try:
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Data preprocessing
        from ml.preprocess import preprocess_login_data
        df = preprocess_login_data(df)
        
        # Train model
        MODEL = train_isolation_forest(df, contamination=contamination)
        
        # Save model
        os.makedirs('model', exist_ok=True)
        save_model(MODEL, MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained from file successfully',
            'samples_trained': len(df),
            'filename': file.filename
        }), 200
        
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
            df = pd.read_csv(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Data preprocessing
        from ml.preprocess import preprocess_login_data
        df = preprocess_login_data(df)
        
        # Prediction
        predictions = predict_anomaly(MODEL, df)
        anomaly_scores = get_anomaly_scores(MODEL, df)
        
        # Add results to dataframe
        df['prediction'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = predictions == -1
        
        # Create statistics
        total_records = len(df)
        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()
        
        # Convert anomalies dataframe - convert numpy types to Python types
        anomalies_list = df[df['is_anomaly']][['is_night', 'login_frequency', 
                                                'location_change', 'device_change', 
                                                'login_result', 'time_delta', 
                                                'anomaly_score']].to_dict('records')
        
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
