# Anomaly Detection System - RFFF Dataset Analysis

System for detecting abnormal behavior based on RFFF Dataset using Isolation Forest model - an unsupervised learning model for anomaly detection.

## Project Structure

```
project/
├── app.py                    # Flask backend main
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── data/
│   └── rfff-dataset.csv     # RFFF Dataset - Login data
├── model/
│   └── isolation_forest.joblib  # Trained model
├── ml/
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   └── predict.py           # Prediction
└── frontend/
    └── index.html           # Web interface
```

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

Application will run at: http://localhost:5000

## Usage

### Web Interface

1. **Train model:**
   - Go to "Training" tab
   - Enter CSV file path (default: `data/rfff-dataset.csv`)
   - Enter expected anomaly rate (default: 0.1)
   - Click "Train Model"

2. **Single Prediction:**
   - Go to "Single Prediction" tab
   - Enter login behavior features
   - Click "Predict"

3. **Batch Prediction:**
   - Go to "Batch Prediction" tab
   - Enter CSV file path
   - Click "Batch Predict"

## API Endpoints

### POST /api/train

Train the model

**Request:**

```json
{
  "csv_file": "data/login_logs.csv",
  "contamination": 0.1
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Model trained and saved successfully",
  "samples_trained": 35
}
```

### POST /api/predict

Predict a single login

**Request:**

```json
{
  "hour": 2,
  "is_night": 1,
  "login_frequency": 5,
  "location_change": 1,
  "device_change": 1,
  "login_result": 0,
  "time_delta": 30
}
```

**Response:**

```json
{
  "status": "success",
  "prediction": -1,
  "anomaly_score": -0.123,
  "is_anomaly": true,
  "timestamp": "2024-01-30T10:30:00.000000"
}
```

### POST /api/predict-batch

Batch prediction

**Request:**

```json
{
  "csv_file": "data/login_logs.csv"
}
```

**Response:**

```json
{
    "status": "success",
    "total_records": 35,
    "anomaly_count": 5,
    "normal_count": 30,
    "anomaly_percentage": 14.29,
    "anomalies": [...]
}
```

### GET /api/status

Check system status

**Response:**

```json
{
  "status": "ok",
  "model_status": "trained",
  "timestamp": "2024-01-30T10:30:00.000000"
}
```

## Features

1. **hour** (0-23): Login hour
2. **is_night** (0/1): Is night-time (22:00-06:00)
3. **login_frequency**: Login frequency in short time period
4. **location_change** (0/1): Geographic/IP location change
5. **device_change** (0/1): Device change
6. **login_result** (0/1): 0 = success, 1 = failure
7. **time_delta**: Time interval between two logins (seconds)

## Prediction Results

- **1**: Normal behavior
- **-1**: Anomalous behavior

## Isolation Forest

Isolation Forest is an unsupervised learning model designed for anomaly detection:

- **Advantages:**
  - Efficient with high-dimensional data
  - Does not require labeled data
  - Handles outliers well
  - Fast and scales with data size

- **Principle:**
  - Builds random decision trees
  - Anomalous points are isolated faster
  - Anomaly score calculated based on average depth in trees

## Notes

- Model is saved as `.joblib` for reuse
- Data is preprocessed before training or prediction
- Contamination ratio affects anomaly detection sensitivity
- Isolation Forest detects anomalies, doesn't confirm attacks

## Future Development

System can be extended with:

- Database integration (MongoDB, PostgreSQL)
- API authentication
- Logging and monitoring
- Cloud deployment (AWS, GCP, Azure)
- Automatic periodic model retraining

## Tác giả

Nguyễn Hoàng Lộc - Đồ án tốt nghiệp

## Giấy phép

MIT
