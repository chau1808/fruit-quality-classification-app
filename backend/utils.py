# backend/utils.py
import torch
from torchvision import transforms
from PIL import Image
import io

# Class names - cập nhật theo dataset thực tế của bạn
CLASS_NAMES = ['average', 'bad', 'good']

def process_image(file_bytes):
    """
    Xử lý ảnh từ bytes thành tensor cho model
    """
    try:
        # Chuyển bytes thành PIL Image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Chuyển sang RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform giống như lúc training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Áp dụng transform và thêm batch dimension
        img_tensor = transform(image).unsqueeze(0)  # Thêm batch dimension
        
        return img_tensor
        
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")
        raise e

def predict_class(output):
    """
    Dự đoán class từ model output
    """
    try:
        # Lấy class có confidence cao nhất
        _, predicted_idx = torch.max(output, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        
        return predicted_class
        
    except Exception as e:
        print(f"❌ Lỗi prediction: {e}")
        raise e

def get_confidence_scores(output):
    """
    Tính confidence scores cho tất cả classes
    """
    try:
        # Softmax để có probability distribution
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Tạo dictionary với class names và scores
        confidence_scores = {
            class_name: float(prob) * 100 
            for class_name, prob in zip(CLASS_NAMES, probabilities)
        }
        
        return confidence_scores
        
    except Exception as e:
        print(f"❌ Lỗi tính confidence: {e}")
        raise e