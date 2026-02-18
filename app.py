import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
UPLOAD_FOLDER = "static/uploads"

# ---- LOCKED BEST SETTINGS (ของม่อน) ----
CRACK_THRESHOLD = 0.58   # ด่านหลัก: crack_max >= 0.58 -> แตก
HIT_THRESHOLD   = 0.48   # ด่านรอง: ต่อ crop
HIT_K           = 2      # ต้องเจออย่างน้อย 2 crop ถึงถือว่าแตก

# ---- Multi-crop ----
USE_MULTI_CROP = True
CROP_RATIO = 0.75
USE_9_CROP = True

# ---- Stone gate (OpenCV) ----
# ปรับให้ "เข้มงวดขึ้น" ถ้าแมว/คนยังผ่าน: เพิ่ม STONE_LAP_MIN และ/หรือ STONE_EDGE_MIN
STONE_LAP_MIN  = 90.0     # แนะนำ 80-140
STONE_EDGE_MIN = 0.015    # แนะนำ 0.01-0.03

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# LOAD MODEL (checkpoint dict)
# ==========================================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()

class_to_idx = ckpt.get("class_to_idx")
if class_to_idx is None:
    raise RuntimeError("best_model.pth ไม่มี class_to_idx — ต้องเทรนด้วย train_best.py เวอร์ชันที่เซฟ dict")

CRACK_NAME = "Crack"
NOCRACK_NAME = "No Crack"
if CRACK_NAME not in class_to_idx or NOCRACK_NAME not in class_to_idx:
    raise RuntimeError(f"class_to_idx ไม่เจอ '{CRACK_NAME}' หรือ '{NOCRACK_NAME}' -> {class_to_idx}")

crack_idx = class_to_idx[CRACK_NAME]
no_crack_idx = class_to_idx[NOCRACK_NAME]

IMG_SIZE = int(ckpt.get("img_size", 300))

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# FLASK
# ==========================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
last_uploaded_path = None


# ===============================
# 🔍 CV Stone Gate
# ===============================
def is_stone_cv(bgr_img):
    """
    return: (is_stone: bool, lap_var: float, edge_density: float)
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # texture
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (bgr_img.shape[0] * bgr_img.shape[1]))

    is_stone = (lap_var >= STONE_LAP_MIN) and (edge_density >= STONE_EDGE_MIN)
    return is_stone, float(lap_var), float(edge_density)


def stone_confidence(lap_var, edge_density):
    """
    ให้ตัวเลขความมั่นใจฝั่ง "เป็นหิน" แบบคร่าวๆ (0-100)
    ไม่ได้ AI จริง แต่ทำให้ UI ดูสมเหตุสมผล
    """
    lap_score = min(1.0, max(0.0, (lap_var - STONE_LAP_MIN) / (STONE_LAP_MIN * 0.8)))
    edge_score = min(1.0, max(0.0, (edge_density - STONE_EDGE_MIN) / (STONE_EDGE_MIN * 1.0)))
    conf = (0.6 * lap_score + 0.4 * edge_score) * 100.0
    return round(conf, 2)


# ===============================
# 🧠 AI Predict
# ===============================
def _predict_probs(pil_img: Image.Image):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return float(probs[crack_idx].item()), float(probs[no_crack_idx].item())


def predict_image_ai(pil_img: Image.Image):
    """
    Returns:
      crack_max (float),
      no_crack_max (float),
      crack_probs (list[float])
    """
    if not USE_MULTI_CROP:
        c, n = _predict_probs(pil_img)
        return c, n, [c]

    W, H = pil_img.size
    crop_size = int(min(W, H) * CROP_RATIO)
    crop_size = max(32, crop_size)

    def crop_box(x, y):
        return (x, y, x + crop_size, y + crop_size)

    boxes = [
        crop_box(0, 0),                                   # TL
        crop_box(W - crop_size, 0),                       # TR
        crop_box(0, H - crop_size),                       # BL
        crop_box(W - crop_size, H - crop_size),           # BR
        crop_box((W - crop_size)//2, (H - crop_size)//2)  # Center
    ]

    if USE_9_CROP:
        boxes += [
            crop_box((W - crop_size)//2, 0),                 # top-center
            crop_box((W - crop_size)//2, H - crop_size),     # bottom-center
            crop_box(0, (H - crop_size)//2),                 # left-center
            crop_box(W - crop_size, (H - crop_size)//2),     # right-center
        ]

    crack_probs = []
    no_probs = []
    for b in boxes:
        patch = pil_img.crop(b)
        c, n = _predict_probs(patch)
        crack_probs.append(c)
        no_probs.append(n)

    return max(crack_probs), max(no_probs), crack_probs


def decide_crack(crack_max, crack_probs):
    crack_hits = sum(p >= HIT_THRESHOLD for p in crack_probs)
    is_crack = (crack_max >= CRACK_THRESHOLD) or (crack_hits >= HIT_K)
    return is_crack, crack_hits


# ===============================
# 🌐 Route หลัก
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    global last_uploaded_path

    original_image = None
    result_image = None
    result_text = None
    confidence = None
    crack = False
    crack_count = 0
    processing_time = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html")

        start_time = time.time()

        # ✅ กันชื่อซ้ำ
        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            ext = ".jpg"

        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(file_path)
        last_uploaded_path = file_path

        # ---- Load for CV gate ----
        bgr = cv2.imread(file_path)
        if bgr is None:
            # อ่านไฟล์ไม่ได้
            result_text = "❌ ไฟล์ภาพไม่ถูกต้อง"
            processing_time = round(time.time() - start_time, 3)
            return render_template("index.html",
                                   result_text=result_text,
                                   confidence=0,
                                   crack=False,
                                   crack_count=0,
                                   processing_time=processing_time)

        # ✅ Stone gate ก่อน
        ok_stone, lap_var, edge_density = is_stone_cv(bgr)
        processing_time = round(time.time() - start_time, 3)

        original_image = url_for("static", filename=f"uploads/{unique_name}")
        result_image = original_image  # เอารูปเดิมแสดงทั้งคู่ (เหมือนเดิม)

        if not ok_stone:
            result_text = "❌ ไม่ใช่หิน"
            confidence = stone_confidence(lap_var, edge_density)  # % ว่า "เป็นหิน" (ต่ำๆ)
            crack = False
            crack_count = 0

            return render_template(
                "index.html",
                original_image=original_image,
                result_image=result_image,
                result_text=result_text,
                confidence=confidence,
                crack=crack,
                crack_count=crack_count,
                processing_time=processing_time
            )

        # ---- AI Crack/No Crack ----
        pil_img = Image.open(file_path).convert("RGB")
        crack_max, no_crack_max, crack_probs = predict_image_ai(pil_img)
        is_crack, crack_hits = decide_crack(crack_max, crack_probs)

        crack = bool(is_crack)
        crack_count = 1 if crack else 0
        confidence = round((crack_max if crack else no_crack_max) * 100, 2)

        if crack:
            result_text = "❌ พบรอยแตก"
        else:
            result_text = "✅ ไม่พบรอยแตก"

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        result_text=result_text,
        confidence=confidence,
        crack=crack,
        crack_count=crack_count,
        processing_time=processing_time
    )


@app.route("/rescan", methods=["POST"])
def rescan():
    global last_uploaded_path

    if not last_uploaded_path or not os.path.exists(last_uploaded_path):
        return jsonify({"confidence": 0, "crack_count": 0, "processing_time": 0, "status": "NO_IMAGE"})

    start_time = time.time()

    bgr = cv2.imread(last_uploaded_path)
    if bgr is None:
        return jsonify({"confidence": 0, "crack_count": 0, "processing_time": 0, "status": "BAD_IMAGE"})

    ok_stone, lap_var, edge_density = is_stone_cv(bgr)
    if not ok_stone:
        return jsonify({
            "confidence": stone_confidence(lap_var, edge_density),
            "crack_count": 0,
            "processing_time": round(time.time() - start_time, 3),
            "status": "NOT_STONE"
        })

    pil_img = Image.open(last_uploaded_path).convert("RGB")
    crack_max, no_crack_max, crack_probs = predict_image_ai(pil_img)
    is_crack, crack_hits = decide_crack(crack_max, crack_probs)

    return jsonify({
        "confidence": round((crack_max if is_crack else no_crack_max) * 100, 2),
        "crack_count": 1 if is_crack else 0,
        "processing_time": round(time.time() - start_time, 3),
        "status": "CRACK" if is_crack else "NO_CRACK"
    })


if __name__ == "__main__":
    print("===================================")
    print("Stone AI Inspection Server Started")
    print("Using device:", DEVICE)
    print("class_to_idx:", class_to_idx)
    print("USE_MULTI_CROP:", USE_MULTI_CROP, "| USE_9_CROP:", USE_9_CROP, "| CROP_RATIO:", CROP_RATIO)
    print("CRACK_THRESHOLD:", CRACK_THRESHOLD, "| HIT_THRESHOLD:", HIT_THRESHOLD, "| HIT_K:", HIT_K)
    print("STONE_LAP_MIN:", STONE_LAP_MIN, "| STONE_EDGE_MIN:", STONE_EDGE_MIN)
    print("===================================")
    app.run(debug=True)
