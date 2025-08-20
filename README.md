# 🍎 Ứng Dụng Phân Loại Chất Lượng Trái Cây

Ứng dụng web sử dụng deep learning để phân loại chất lượng trái cây thành hai loại: **Tốt (Good)** và **Kém (Bad)**. Được xây dựng với Flask (backend) và React (frontend), sử dụng mô hình VGG16 đã được fine-tuned.

## 🚀 Tính Năng

- **Phân Loại Thời Gian Thực**: Tải ảnh lên hoặc sử dụng camera để chụp ảnh trái cây
- **Deep Learning**: Mô hình VGG16 được fine-tune trên dataset chất lượng trái cây
- **Giao Diện Web**: Frontend React hiện đại với thiết kế responsive
- **REST API**: Backend Flask với inference model PyTorch
- **Hai Lớp Chất Lượng**: Phân loại Tốt, Kém

## 🛠 Công Nghệ Sử Dụng

**Backend:**
- Python
- Flask
- PyTorch
- VGG16 (pre-trained trên ImageNet)

**Frontend:**
- React
- TailwindCSS
- Modern JavaScript (ES6+)

## 📋 Yêu Cầu Hệ Thống

- Python 3.8+
- Node.js 14+
- npm hoặc yarn
- Git

## 🔧 Cài Đặt & Thiết Lập

### 1. Clone Repository

Mở terminal và clone repository về máy của bạn:

```bash
git clone https://github.com/chau1808/fruit-quality-classification-app.git
cd fruit-quality-classification-app
```

### 2. Backend (Flask + PyTorch)

Để cài đặt và chạy backend, hãy làm theo các bước dưới đây:

Mở terminal, điều hướng đến thư mục **backend** và tạo môi trường ảo:

```bash
cd backend
python -m venv venv
```

Kích hoạt môi trường ảo:
- **Windows:** `venv\Scripts\activate`
- **Linux/Mac:** `source venv/bin/activate`

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

**Tải model từ Google Drive:**
- 👉 [Link tải model tại đây](https://drive.google.com/file/d/1qjqV3ZvhPIbfqd3ShcSDSNTuGjC9iNLl/view?usp=drive_link)
- Sau khi tải xong, hãy đặt file **vgg16_fruit_model_2cls.pth** vào thư mục:

```bash
backend/model/vgg16_fruit_model_2cls.pth
```

Chạy backend:

```bash
python app.py
```

Backend mặc định chạy tại: `http://127.0.0.1:5000`

### 3. Frontend (React)

Mở một terminal mới, điều hướng đến thư mục **frontend**:

```bash
cd ../frontend
```

Cài đặt các dependencies và chạy frontend:

```bash
npm install
npm start
```

Frontend mặc định chạy tại: `http://localhost:3000`

## 🎯 Cách Sử Dụng

1. Mở frontend tại: `http://localhost:3000`
2. Tải ảnh lên hoặc sử dụng camera để chụp ảnh
3. Ảnh sẽ được gửi đến **Flask API backend**
4. Backend sử dụng model **VGG16** đã huấn luyện để dự đoán
5. Kết quả dự đoán sẽ hiển thị trên giao diện

## 🔄 Huấn Luyện Lại Model (Tùy Chọn)

Nếu bạn muốn huấn luyện lại mô hình thay vì dùng model có sẵn, hãy làm theo các bước sau:

1. Mở file notebook: **vgg16_fruit_model_2cls.ipynb** (khuyến khích chạy trên Google Colab hoặc Jupyter)

2. Chuẩn bị dataset (ví dụ từ Kaggle). Dataset cần có cấu trúc:

```
dataset/
├── train/
│   ├── good/
│   └── bad/
├── val/
│   ├── good/
│   └── bad/
└── test/
    ├── good/
    └── bad/
```

3. Notebook sẽ:
   - Load dataset và thay đổi kích thước ảnh về 224x224
   - Áp dụng **augmentation** (flip, rotate, normalize...)
   - Sử dụng **VGG16 pretrained** trên ImageNet
   - Fine-tune các layer cuối cho 2 lớp (`good`, `bad`)
   - Lưu model sau khi train:

```python
torch.save(model.state_dict(), "vgg16_fruit_model_2cls.pth")
```

4. Sau khi huấn luyện xong, copy model vào:

```bash
backend/model/vgg16_fruit_model_2cls.pth
```

## 📁 Cấu Trúc Dự Án

```
fruit-quality-classification-app/
├── backend/
│   ├── model/
│   │   └── vgg16_fruit_model_2cls.pth
│   ├── app.py
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
├── training
└── README.md
```

## 📝 API Endpoints

### POST `/predict`
- **Mô tả**: Phân loại chất lượng trái cây từ ảnh được tải lên
- **Content-Type**: `multipart/form-data`
- **Tham số**: `file` (file ảnh)
- **Phản hồi**: JSON với kết quả dự đoán

```json
{
  "prediction": "good",
  "confidence": 0.87,
  "probabilities": {
    "good": 0.87,
    "bad": 0.13
  }
}
```

## 🔍 Chi Tiết Model

- **Kiến trúc**: VGG16 (pre-trained trên ImageNet)
- **Fine-tuning**: Các layer cuối được điều chỉnh cho phân loại 2 lớp chất lượng trái cây
- **Kích thước đầu vào**: 224x224x3
- **Các lớp**: Good (Tốt), Bad (Kém)
- **Framework**: PyTorch

## 📌 Ghi Chú Quan Trọng

- File model `.pth` không được lưu trong repo vì GitHub giới hạn 100MB. Bạn cần tải model từ Google Drive hoặc huấn luyện lại bằng notebook
- Nếu huấn luyện lại, nên bật **GPU** (trên Colab: Runtime → Change runtime type → GPU)
- Đảm bảo cả backend và frontend đều đang chạy để ứng dụng hoạt động đầy đủ

## 🚦 Xử Lý Sự Cố

**Vấn đề Backend:**
- Đảm bảo môi trường ảo đã được kích hoạt
- Kiểm tra tất cả dependencies đã được cài đặt: `pip install -r requirements.txt`
- Kiểm tra file model có tồn tại trong `backend/model/vgg16_fruit_model_2cls.pth`

**Vấn đề Frontend:**
- Xóa cache npm: `npm cache clean --force`
- Xóa `node_modules` và cài đặt lại: `rm -rf node_modules && npm install`
- Kiểm tra backend có chạy tại `http://127.0.0.1:5000`

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/TinhNangMoi`)
3. Commit thay đổi (`git commit -m 'Thêm tính năng mới'`)
4. Push lên branch (`git push origin feature/TinhNangMoi`)
5. Mở Pull Request

## 📄 Giấy Phép

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Tác Giả

- **Châu** - *Công việc ban đầu* - [chau1808](https://github.com/chau1808)

## 🙏 Lời Cảm Ơn

- Mô hình VGG16 từ torchvision của PyTorch
- Pre-trained weights từ ImageNet
- Những người đóng góp dataset chất lượng trái cây

## 💡 Lưu Ý Thêm

- Ứng dụng hỗ trợ các định dạng ảnh phổ biến: JPG, PNG, JPEG
- Để có kết quả tốt nhất, hãy sử dụng ảnh có độ phân giải cao và rõ nét
- Model đã được huấn luyện trên nhiều loại trái cây khác nhau
- Thời gian xử lý phụ thuộc vào kích thước ảnh và cấu hình máy

## 🔗 Links Hữu Ích

- [PyTorch Documentation](https://pytorch.org/docs/)
- [React Documentation](https://reactjs.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
