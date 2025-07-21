# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from utils import process_image, predict_class  # import từ utils.py
import os

app = Flask(__name__)
CORS(app)  # Cho phép frontend kết nối

# Cấu hình
MODEL_PATH = 'model/vgg16_fruit_model.pth'
CLASS_NAMES = ['average', 'bad', 'good']  # Thay đổi theo class names thực tế của bạn
NUM_CLASSES = len(CLASS_NAMES)

def load_model():
    """Load VGG16 model với weights đã train"""
    try:
        print("🔄 Đang tải model...")
        
        # Tạo lại model architecture giống như lúc training
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Đóng băng features
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Thay classifier
        model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
        
        # Load trained weights
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            print(f"✅ Đã load model từ: {MODEL_PATH}")
        else:
            print(f"⚠️ Không tìm thấy model tại: {MODEL_PATH}")
            print("🔄 Sử dụng model chưa train (chỉ để test)")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return None

# Load model khi khởi động app
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kiểm tra model
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Kiểm tra file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Xử lý ảnh
        file_bytes = file.read()
        img_tensor = process_image(file_bytes)
        
        # Dự đoán
        with torch.no_grad():
            output = model(img_tensor)
            prediction = predict_class(output)
            
            # Tính confidence scores
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence_scores = {
                class_name: float(prob) * 100 
                for class_name, prob in zip(CLASS_NAMES, probabilities)
            }
        
        # Tính confidence cho class được predict
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        max_confidence = float(torch.max(probabilities).item())
        
        return jsonify({
            'success': True,
            'class': prediction,
            'confidence': max_confidence,  # Frontend expect 'confidence' (0-1)
            'confidence_scores': confidence_scores,
            'message': f'Predicted class: {prediction}'
        })
        
    except Exception as e:
        print(f"❌ Lỗi prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fruit Quality Classification API',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        },
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print(f"📊 Classes: {CLASS_NAMES}")
    print(f"🤖 Model loaded: {model is not None}")
    app.run(debug=True, host='0.0.0.0', port=5000)