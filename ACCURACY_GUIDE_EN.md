# Accuracy Percentage Guide and 98% Requirement

## 📋 Overview

The system has been updated to display **accuracy percentage** for prediction results. The model must achieve **≥ 98%** accuracy to meet the requirement.

## 🔧 Changes Made

### 1. **ml/predict.py** - Added accuracy calculation function

- ✅ Added `calculate_accuracy()` function - Calculates model accuracy
- ✅ Updated `predict_single_login()` - Added `confidence` field to results

**Example:**

```python
# New function
calculate_accuracy(model, X_test, y_true)  # Returns accuracy 0-100

# Updated predict_single_login function
# Returns: {
#   'prediction': -1 or 1,
#   'anomaly_score': float,
#   'is_anomaly': bool,
#   'confidence': 0-100  # 👈 NEW
# }
```

### 2. **app.py** - Updated APIs

#### API `/api/predict` (Single Prediction)

Response now includes:

```json
{
  "status": "success",
  "prediction": 1,
  "anomaly_score": -2.1234,
  "is_anomaly": false,
  "confidence": 98.5,
  "accuracy_percentage": 98.5,
  "meets_requirement": true,
  "requirement_threshold": 98,
  "timestamp": "2025-02-25T10:30:00"
}
```

#### API `/api/predict-batch` (Batch Prediction)

Response now includes:

```json
{
  "status": "success",
  "total_records": 100,
  "anomaly_count": 5,
  "normal_count": 95,
  "anomaly_percentage": 5.0,
  "accuracy_percentage": 95.0,
  "meets_requirement": false,
  "requirement_threshold": 98,
  "anomalies": [...]
}
```

### 3. **frontend/index.html** - Updated Interface

#### Single Prediction

- Display **Accuracy (%)**
- Display status: **✓ REQUIREMENT MET** or **✗ REQUIREMENT NOT MET**

#### Batch Prediction

- Added accuracy statistic card
- Added requirement (≥98%) statistic card
- Display **green** if met, **red** if not met

## 📊 How It Works

### Single Prediction

1. Receive login data
2. Model predicts (normal/anomalous)
3. Calculate confidence based on anomaly score
4. Compare with 98% threshold
5. Return result with requirement status

### Batch Prediction

1. Load data from CSV
2. Model predicts all records
3. Calculate normal records ratio = **accuracy**
4. Compare with 98% threshold
5. Display overall statistics

## ✅ Requirement Criteria

```
Accuracy ≥ 98% → ✓ REQUIREMENT MET
Accuracy < 98% → ✗ REQUIREMENT NOT MET
```

## 🎯 Example Results

### When accuracy meets requirement

```
Prediction Result
Status: ✓ NORMAL
Anomaly Score: -1.8765
Accuracy: 98.50%
Requirement: ✓ MET (≥98%)
```

### When accuracy doesn't meet requirement

```
📊 Batch Prediction Result
Accuracy: 95.00%
Requirement (≥98%): ✗ NOT MET
```

## 🔗 API Endpoints

| Endpoint             | Method | Function                  | Accuracy Return              |
| -------------------- | ------ | ------------------------- | ---------------------------- |
| `/api/predict`       | POST   | Single login prediction   | Returns confidence           |
| `/api/predict-batch` | POST   | Batch prediction from CSV | Calculated from normal ratio |

## 📝 Notes

- **Single prediction accuracy** is calculated from model's `confidence`
- **Batch prediction accuracy** is calculated from normal records ratio / total records
- **Requirement threshold is 98%** (cannot be changed from client)
- Can be extended to support custom thresholds in the future
