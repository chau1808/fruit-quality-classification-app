# utils.py
import io, torch
from PIL import Image
from torchvision import transforms

# Chuẩn hóa theo ImageNet (bắt buộc cho VGG16 pretrain)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),     # BẮT BUỘC để đồng nhất kích thước
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def process_image(file_bytes, transform=BASE_TRANSFORM):
    """
    bytes → tensor shape (1,3,224,224)
    """
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return transform(img).unsqueeze(0)

def predict_class(output, class_names):
    """
    output: torch.Tensor shape [1, C]
    class_names: list[str]
    """
    _, idx = torch.max(output, 1)
    return class_names[idx.item()]

def get_confidence_scores(output, class_names):
    """
    output: torch.Tensor shape [1, C]
    return: {class: prob%}
    """
    probs = torch.softmax(output[0], dim=0)
    return {cls: round(float(p) * 100, 2) for cls, p in zip(class_names, probs)}
