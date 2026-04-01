# Hướng dẫn sử dụng chức năng Upload File CSV

## 🎯 Tính năng mới

Hệ thống hiện hỗ trợ 2 cách để tải file CSV:

### 1. **Từ đường dẫn (File Path)**

- Nhập đường dẫn tuyệt đối hoặc tương đối đến file CSV
- Ví dụ: `data/login_logs.csv` hoặc `C:\data\logs.csv`
- Phù hợp khi file nằm trên máy chủ

### 2. **Upload từ máy tính (File Upload)** ✨ NEW

- Chọn file CSV trực tiếp từ máy tính
- Không cần biết đường dẫn đầy đủ
- Dễ dàng và an toàn hơn
- Hỗ trợ kéo thả (drag & drop)

## 📝 Các mục sử dụng Upload

### Tab "Huấn luyện"

- Có 2 nút toggle: "📂 Từ đường dẫn" và "📤 Upload từ máy"
- Chuyển giữa 2 chế độ bằng cách nhấn nút toggle
- Upload mode: Chọn file CSV từ máy, đặt tỷ lệ bất thường, nhấn "🚀 Huấn luyện từ file"

### Tab "Dự đoán hàng loạt"

- Tương tự tab "Huấn luyện"
- Có 2 chế độ: từ đường dẫn hoặc upload
- Upload mode: Chọn file CSV từ máy, nhấn "📊 Dự đoán hàng loạt"

## 📤 Cách sử dụng Upload

### Cách 1: Nhấp để chọn

1. Nhấn vào ô "File input"
2. Chọn file CSV từ máy tính
3. File sẽ hiển thị: "✓ File được chọn: tên_file.csv"
4. Nhấn nút xử lý (Huấn luyện hoặc Dự đoán)

### Cách 2: Kéo thả (Drag & Drop)

1. Kéo file CSV từ File Explorer
2. Thả vào ô "File input"
3. File sẽ tự động được chọn
4. Nhấn nút xử lý

## 🔧 Yêu cầu File

- **Định dạng**: CSV (.csv)
- **Dung lượng tối đa**: 16 MB
- **Cột yêu cầu**:
  - `hour` (0-23)
  - `is_night` (0/1)
  - `login_frequency` (số nguyên)
  - `location_change` (0/1)
  - `device_change` (0/1)
  - `login_result` (0/1)
  - `time_delta` (số)

## 📂 Backend API Endpoints

### POST /api/train-upload

Upload file CSV để huấn luyện mô hình

**Request:**

```
POST /api/train-upload
Content-Type: multipart/form-data

file: <file CSV>
contamination: 0.1
```

**Response:**

```json
{
  "status": "success",
  "message": "Mô hình đã huấn luyện từ file thành công",
  "samples_trained": 35,
  "filename": "login_logs.csv"
}
```

### POST /api/predict-batch-upload

Upload file CSV để dự đoán hàng loạt

**Request:**

```
POST /api/predict-batch-upload
Content-Type: multipart/form-data

file: <file CSV>
```

**Response:**

```json
{
    "status": "success",
    "filename": "login_logs.csv",
    "total_records": 35,
    "anomaly_count": 5,
    "normal_count": 30,
    "anomaly_percentage": 14.29,
    "anomalies": [...]
}
```

## 🚀 Cách chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
python app.py

# Truy cập
http://localhost:5000
```

## ⚠️ Lưu ý

- File sẽ được lưu tạm ở thư mục `uploads/`
- Dữ liệu upload không được lưu lâu dài (chỉ để xử lý)
- Mô hình đã huấn luyện sẽ được lưu ở `model/isolation_forest.joblib`
- Kiểm tra file CSV có format đúng trước khi upload

## 🐛 Troubleshooting

### Lỗi "Chỉ chấp nhận file CSV"

- Kiểm tra file có phần mở rộng `.csv`
- Không upload file Excel hay định dạng khác

### Lỗi "Lỗi đọc file CSV"

- Kiểm tra định dạng CSV (dấu phân tách, encoding)
- Đảm bảo file không bị hỏng
- Mở file bằng text editor để kiểm tra format

### Upload quá lâu

- File CSV quá lớn (> 16MB)
- Hãy chia nhỏ file hoặc giảm số lượng bản ghi

## 📊 Ví dụ File CSV

```csv
hour,is_night,login_frequency,location_change,device_change,login_result,time_delta
14,0,1,0,0,0,120
2,1,3,1,1,0,45
15,0,1,0,0,0,130
3,1,5,1,1,1,20
16,0,2,0,0,0,110
```

Chúc bạn sử dụng thành công! 🎉
