# app.py
import os, json, io, time, base64
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from PIL import Image

from utils import process_image, predict_class, get_confidence_scores

app = Flask(__name__)
CORS(app)

# ====== Paths (có thể override bằng biến môi trường) ======
MODEL_PATH   = os.getenv("MODEL_PATH",   "model/vgg16_fruit_model_2cls.pth")
CLASSES_JSON = os.getenv("CLASSES_JSON", "model/classes.json")

# ====== Meta / cấu hình trả về ======
MODEL_META = {
    "arch": "vgg16",
    "version": os.getenv("MODEL_VERSION", "v1.0.0")
}
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "70.0"))  # % cho cảnh báo
TOPK = int(os.getenv("TOPK", "3"))

# ====== Load classes.json (hỗ trợ nhiều format) ======
def load_class_names(default=("bad_fruit","good_fruit")):
    """
    Hỗ trợ:
    1) {"classes": ["bad_fruit","good_fruit"]}
    2) ["bad_fruit","good_fruit"]
    3) {"0":"bad_fruit","1":"good_fruit"} (mapping index->name)
    """
    try:
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
            classes = data["classes"]
        elif isinstance(data, list):
            classes = data
        elif isinstance(data, dict):  # mapping index->name
            try:
                keys = sorted(data.keys(), key=lambda k: int(k))
            except Exception:
                keys = sorted(data.keys())
            classes = [data[k] for k in keys]
        else:
            raise ValueError("Định dạng classes.json không hợp lệ.")

        if not isinstance(classes, list) or len(classes) < 2:
            raise ValueError("Danh sách lớp không hợp lệ.")
        return classes

    except Exception as e:
        print(f"⚠️ Không đọc được classes.json, dùng mặc định: {e}")
        return list(default)

CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# ====== Build & Load model ======
def build_model(num_classes: int) -> torch.nn.Module:
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def _extract_state_dict(state):
    """
    Trả về state_dict thuần từ nhiều kiểu checkpoint:
    - state là dict các weights
    - state có key 'state_dict' hoặc 'model_state_dict'
    - keys có prefix '_orig_mod.' (khi dùng torch.compile) -> strip
    """
    if isinstance(state, dict):
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    return state

def load_model():
    try:
        print("🔄 Loading VGG16...")
        model = build_model(NUM_CLASSES)
        if os.path.exists(MODEL_PATH):
            raw = torch.load(MODEL_PATH, map_location="cpu")
            state = _extract_state_dict(raw)
            try:
                model.load_state_dict(state, strict=True)
            except Exception as e:
                print(f"⚠️ strict=True fail: {e} → thử strict=False")
                model.load_state_dict(state, strict=False)
            print(f"✅ Loaded weights: {MODEL_PATH}")
        else:
            print(f"⚠️ Không tìm thấy model tại: {MODEL_PATH} (hãy train trước)")

        model.eval()
        return model
    except Exception as e:
        print("❌ Lỗi load model:", e)
        return None

model = load_model()

# ====== Endpoints ======
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fruit Quality Classification API (VGG16)",
        "classes": CLASS_NAMES,
        "endpoints": {
            "health":  "GET  /health",
            "labels":  "GET  /labels",
            "predict": "POST /predict  form-data: file=<image>"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "model_path": MODEL_PATH,
        "model_meta": MODEL_META
    })

@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"success": False, "error": "Model not loaded"}), 500

        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Đọc bytes 1 lần để lấy kích thước gốc + tiền xử lý
        raw_bytes = file.read()

        # Kích thước ảnh gốc để hiển thị ở FE
        try:
            _img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            orig_w, orig_h = _img.size
        except Exception:
            orig_w, orig_h = None, None

        # Tiền xử lý & suy luận (đo thời gian)
        t0 = time.time()
        img_tensor = process_image(raw_bytes)  # (1,3,224,224)

        with torch.no_grad():
            output = model(img_tensor)  # [1, C]
            pred = predict_class(output, CLASS_NAMES)  # str
            scores_pct = get_confidence_scores(output, CLASS_NAMES)  # {label: %}

            # Softmax prob 0–1 cho Top-K
            probs = torch.softmax(output[0], dim=0)  # [C]
            k = min(TOPK, NUM_CLASSES)
            topk_prob, topk_idx = torch.topk(probs, k)
            top_k = [
                {
                    "class": CLASS_NAMES[i.item()],
                    "prob": float(p.item()),
                    "pct": round(float(p.item()) * 100, 2)
                }
                for p, i in zip(topk_prob, topk_idx)
            ]

            # Prob/pct của lớp dự đoán
            pred_idx = CLASS_NAMES.index(pred)
            pred_prob = float(probs[pred_idx].item())
            pred_pct  = float(scores_pct.get(pred, 0.0))

        inference_ms = round((time.time() - t0) * 1000, 2)
        threshold_met = pred_pct >= DEFAULT_THRESHOLD
        note = None if threshold_met else "Độ tin cậy thấp, hãy thử ảnh rõ/đủ sáng hơn."

        # ====== Tạo ảnh preview 224x224 (DENORMALIZE) trả về base64 ======
        # utils.process_image đã Normalize theo ImageNet -> cần khử Normalize
        try:
            # img_tensor: [1, 3, 224, 224]
            img = img_tensor[0].cpu().clone()  # [3, 224, 224]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img = img * std + mean                            # de-normalize về [0,1]
            img = torch.clamp(img, 0.0, 1.0)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")  # [H,W,3] uint8
            pil_preview = Image.fromarray(img_np)
            buf = io.BytesIO()
            pil_preview.save(buf, format="PNG")
            preview_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            preview_base64 = None

        return jsonify({
            "success": True,

            # Thông tin dự đoán
            "prediction": {
                "class": pred,
                "prob": round(pred_prob, 6),   # 0–1
                "pct": round(pred_pct, 2)      # 0–100
            },
            "confidence_scores": scores_pct,   # {label: %}
            "top_k": top_k,

            # Ngưỡng cảnh báo
            "threshold": {
                "value_pct": DEFAULT_THRESHOLD,
                "met": threshold_met,
                "note": note
            },

            # Thời gian & input
            "timings": {
                "inference_ms": inference_ms
            },
            "input": {
                "original_size": {"w": orig_w, "h": orig_h},
                "preprocessed_size": {"w": 224, "h": 224}
            },

            # Meta model
            "model": MODEL_META,

            # Ảnh 224x224 sau tiền xử lý (đã de-normalize)
            "preview": preview_base64
        })

    except Exception as e:
        print("❌ Predict error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting server...")
    print(f"📦 MODEL_PATH   = {MODEL_PATH}")
    print(f"📦 CLASSES_JSON = {CLASSES_JSON}")
    print(f"📊 Classes: {CLASS_NAMES}")
    print(f"🤖 Model loaded: {model is not None}")
    app.run(host="0.0.0.0", port=5000, debug=True)
