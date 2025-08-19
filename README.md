# 🍎 Fruit Quality Classification App

Ứng dụng phân loại chất lượng trái cây (good / average / bad) sử dụng **VGG16 + Transfer Learning**.  
Gồm 2 phần: **Frontend (React)** và **Backend (Flask + PyTorch)**.  
Model được lưu trên **Google Drive** để tránh file nặng >100MB trong GitHub.

---

## 📂 Cấu trúc thư mục

fruit-quality-classification-app/
│── backend/
│ │── app.py # Flask API
│ │── model/ # Thư mục chứa model (đặt file .pth ở đây)
│ │── requirements.txt # Thư viện backend
│── frontend/
│ │── src/ # Code React frontend
│ │── package.json # Cấu hình npm
│── README.md

yaml
Sao chép
Chỉnh sửa

---

## ⚙️ Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/chau1808/fruit-quality-classification-app.git
cd fruit-quality-classification-app
2. Backend (Flask + PyTorch)
bash
Sao chép
Chỉnh sửa
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
🔹 Tải model từ Google Drive:
👉 Link tải model tại đây
(Sau khi tải, đặt file model vào thư mục:)

bash
Sao chép
Chỉnh sửa
backend/model/vgg16_fruit_model.pth
Chạy backend:

bash
Sao chép
Chỉnh sửa
python app.py
Backend mặc định chạy tại: http://127.0.0.1:5000

3. Frontend (React)
bash
Sao chép
Chỉnh sửa
cd ../frontend
npm install
npm start
Frontend mặc định chạy tại: http://localhost:3000

🚀 Cách sử dụng
Mở frontend tại: http://localhost:3000

Upload ảnh hoặc chụp từ camera.

Ảnh được gửi đến Flask API backend.

Backend sử dụng model VGG16 đã huấn luyện để dự đoán.

Kết quả hiển thị trên giao diện.

📌 Ghi chú
File model .pth không được lưu trong repo vì GitHub giới hạn 100MB.

Bạn cần tải model từ Google Drive trước khi chạy backend.

Nếu muốn huấn luyện lại model: xem notebook fruits_quality_vgg16_training.ipynb.

🛠 Công nghệ
Backend: Python, Flask, PyTorch

Frontend: React, TailwindCSS

Model: VGG16 + Transfer Learning

