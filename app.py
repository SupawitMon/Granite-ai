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

import gdown  # pip install gdown

# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model download (Google Drive) ----
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "15dY4OBZ_pii_NR8FnRpESjIpZ8omsXtH"

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
STONE_LAP_MIN  = 90.0     # 80-140
STONE_EDGE_MIN = 0.015    # 0.01-0.03

# ---- Allowed extensions ----
# NOTE: อนุญาต gif ให้ "อัปได้" แต่จะตอบว่า "ยังไม่รองรับ GIF" ตามที่ม่อนต้องการ
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# Download model if missing
# ==========================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

ensure_model()

# ==========================================
# LOAD MODEL (checkpoint dict)
# ==========================================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# รองรับทั้งแบบ dict (มี state_dict) และแบบ state_dict ตรงๆ
state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

class_to_idx = ckpt.get("class_to_idx") if isinstance(ckpt, dict) else None
if class_to_idx is None:
    raise RuntimeError("best_model.pth ไม่มี class_to_idx — กรุณาเซฟโมเดลเป็น dict ที่มี class_to_idx")

CRACK_NAME = "Crack"
NOCRACK_NAME = "No Crack"
if CRACK_NAME not in class_to_idx or NOCRACK_NAME not in class_to_idx:
    raise RuntimeError(f"class_to_idx ไม่เจอ '{CRACK_NAME}' หรือ '{NOCRACK_NAME}' -> {class_to_idx}")

crack_idx = class_to_idx[CRACK_NAME]
no_crack_idx = class_to_idx[NOCRACK_NAME]

IMG_SIZE = int(ckpt.get("img_size", 300)) if isinstance(ckpt, dict) else 300

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
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / (bgr_img.shape[0] * bgr_img.shape[1]))

    is_stone = (lap_var >= STONE_LAP_MIN) and (edge_density >= STONE_EDGE_MIN)
    return is_stone, float(lap_var), float(edge_density)


def stone_confidence(lap_var, edge_density):
    lap_score  = min(1.0, max(0.0, (lap_var - STONE_LAP_MIN) / (STONE_LAP_MIN * 0.8)))
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
    if not USE_MULTI_CROP:
        c, n = _predict_probs(pil_img)
        return c, n, [c]

    W, H = pil_img.size
    crop_size = int(min(W, H) * CROP_RATIO)
    crop_size = max(32, crop_size)

    def crop_box(x, y):
        return (x, y, x + crop_size, y + crop_size)

    boxes = [
        crop_box(0, 0),
        crop_box(W - crop_size, 0),
        crop_box(0, H - crop_size),
        crop_box(W - crop_size, H - crop_size),
        crop_box((W - crop_size)//2, (H - crop_size)//2),
    ]

    if USE_9_CROP:
        boxes += [
            crop_box((W - crop_size)//2, 0),
            crop_box((W - crop_size)//2, H - crop_size),
            crop_box(0, (H - crop_size)//2),
            crop_box(W - crop_size, (H - crop_size)//2),
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
# Helpers
# ===============================
def allowed_file_ext(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT, ext


def save_upload(file):
    ok, ext = allowed_file_ext(file.filename)
    if not ok:
        return None, None, "BAD_EXT"

    # ถ้าเป็น gif -> รับไฟล์ได้ แต่ให้ขึ้นข้อความเฉพาะ
    if ext == ".gif":
        return None, None, "GIF_NOT_ALLOWED"

    # กันชื่อซ้ำ + บังคับนามสกุลให้เป็นรูปทั่วไป
    ext = ext if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] else ".jpg"

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(file_path)
    return file_path, unique_name, "OK"


# ===============================
# 🌐 Routes
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

        file_path, unique_name, status = save_upload(file)
        processing_time = round(time.time() - start_time, 3)

        # ---- handle upload status ----
        if status == "GIF_NOT_ALLOWED":
            return render_template(
                "index.html",
                result_text="❌ ยังไม่รองรับไฟล์ GIF",
                status="GIF",
                confidence=0,
                crack=False,
                crack_count=0,
                processing_time=processing_time,
                original_image=None,
                result_image=None
            )

        if status == "BAD_EXT" or file_path is None:
            return render_template(
                "index.html",
                result_text="❌ ไฟล์ภาพไม่ถูกต้อง",
                status="BAD_IMAGE",
                confidence=0,
                crack=False,
                crack_count=0,
                processing_time=processing_time,
                original_image=None,
                result_image=None
            )

        last_uploaded_path = file_path
        original_image = url_for("static", filename=f"uploads/{unique_name}")
        result_image = original_image

        # ---- CV gate ----
        bgr = cv2.imread(file_path)
        if bgr is None:
            return render_template(
                "index.html",
                result_text="❌ ไฟล์ภาพไม่ถูกต้อง",
                status="BAD_IMAGE",
                confidence=0,
                crack=False,
                crack_count=0,
                processing_time=processing_time,
                original_image=original_image,
                result_image=result_image
            )

        ok_stone, lap_var, edge_density = is_stone_cv(bgr)

        if not ok_stone:
            result_text = "❌ ไม่ใช่หิน"
            confidence = stone_confidence(lap_var, edge_density)
            crack = False
            crack_count = 0

            return render_template(
                "index.html",
                original_image=original_image,
                result_image=result_image,
                result_text=result_text,
                status="NOT_STONE",
                confidence=confidence,
                crack=crack,
                crack_count=crack_count,
                processing_time=processing_time
            )

        # ---- AI crack ----
        pil_img = Image.open(file_path).convert("RGB")
        crack_max, no_crack_max, crack_probs = predict_image_ai(pil_img)
        is_crack, crack_hits = decide_crack(crack_max, crack_probs)

        crack = bool(is_crack)
        crack_count = 1 if crack else 0
        confidence = round((crack_max if crack else no_crack_max) * 100, 2)
        result_text = "❌ พบรอยแตก" if crack else "✅ ไม่พบรอยแตก"

        return render_template(
            "index.html",
            original_image=original_image,
            result_image=result_image,
            result_text=result_text,
            status="CRACK" if crack else "NO_CRACK",
            confidence=confidence,
            crack=crack,
            crack_count=crack_count,
            processing_time=processing_time
        )

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        result_text=result_text,
        status=None,
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

    # สำคัญตอน deploy: ให้ใช้ PORT จาก environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
