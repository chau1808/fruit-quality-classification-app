# train.py - Sửa lỗi NameError: name 'model' is not defined

# ✅ Huấn luyện mô hình phân loại chất lượng giỏ trái cây (good, average, bad)
# ✅ Transfer Learning sử dụng VGG16 + PyTorch

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time
from torchvision.models import VGG16_Weights

# ✅ 1. Thiết lập cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "clean_dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 10
MODEL_PATH = "model/vgg16_fruit_model.pth"

# ✅ 2. Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
NUM_CLASSES = len(class_names)  # 🔥 Tự động đếm số lớp theo class folder
print(f"📁 Số lớp phát hiện: {NUM_CLASSES}")
print(f"📁 Tên các lớp: {class_names}")

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 3. Load VGG16 pre-trained + fine-tune
print("🔄 Đang tải VGG16 pre-trained model...")
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# Đóng băng các layer convolutional (không huấn luyện lại)
for param in model.features.parameters():
    param.requires_grad = False

# Thay đổi lớp FC cuối để phù hợp số lớp đầu ra
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
model = model.to(DEVICE)

print(f"✅ Model đã được chuyển sang: {DEVICE}")

# ✅ 4. Cấu hình huấn luyện
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# ✅ 5. Hàm huấn luyện
def train_model(model, loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - Acc: {acc:.2f}%")

    return model

# ✅ 6. Huấn luyện mô hình
print("🚀 Bắt đầu huấn luyện...")
start = time.time()
model = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
end = time.time()
print(f"\n⏱️ Thời gian huấn luyện: {(end - start):.2f} giây")

# ✅ 7. Lưu mô hình
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Đã lưu mô hình vào: {MODEL_PATH}")

# ✅ 8. Đánh giá nhanh
print("🧪 Đánh giá mô hình...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))