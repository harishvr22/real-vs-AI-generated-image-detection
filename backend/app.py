import os
import json
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime

# Try import tensorflow/keras - provide friendly error if missing
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
CLASSES_JSON = os.path.join(MODEL_DIR, "classes.json")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "history.db")

# Ensure database directory exists
os.makedirs(DB_DIR, exist_ok=True)

# ---------- FLASK SETUP ----------
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

model = None
classes = None

# ---------- DATABASE HELPERS ----------
def init_db():
    """Create database and table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT,
            image_name TEXT,
            confidence REAL,
            label TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_history(roll_number, image_name, confidence, label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history (roll_number, image_name, confidence, label, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (roll_number, image_name, confidence, label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT roll_number, image_name, confidence, label, timestamp
        FROM history
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "roll_number": r[0],
            "image_name": r[1],
            "confidence": r[2],
            "label": r[3],
            "timestamp": r[4]
        }
        for r in rows
    ]

# ---------- MODEL LOADING ----------
def load_resources():
    global model, classes
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available. Install tensorflow before running the backend.")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found. Ensure {MODEL_PATH} exists.")
    
    model = load_model(MODEL_PATH)
    
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, 'r') as f:
            inv_map = json.load(f)
            classes = {int(k): v for k, v in inv_map.items()}
    else:
        classes = {0: 'fake', 1: 'real'}
    
    print("✅ Model and classes loaded:", classes)

# ---------- IMAGE PREPROCESS ----------
def preprocess_image(image_bytes, target_size=(128, 128)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------- ROUTES ----------
@app.route('/api/predict', methods=['POST'])
def predict():
    global model, classes
    try:
        # Load model if not already
        if model is None:
            load_resources()

        if 'file' not in request.files:
            return jsonify({"error": "No file sent"}), 400
        
        f = request.files['file']
        roll_number = request.form.get('roll_number', '711523BAM022')
        img_bytes = f.read()

        inp = preprocess_image(img_bytes)
        pred = float(model.predict(inp)[0][0])
        label_idx = 1 if pred >= 0.5 else 0
        label_name = classes.get(label_idx, "real" if label_idx == 1 else "fake")
        confidence = pred if label_idx == 1 else 1 - pred

        # Save to DB
        insert_history(roll_number, f.filename, round(confidence * 100, 2), label_name)

        print(f"✅ Prediction: {label_name} ({confidence * 100:.2f}%)")

        return jsonify({
            "label": label_name,
            "confidence": round(float(confidence), 4),
            "message": "Prediction stored successfully"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        data = fetch_history()
        return jsonify(data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/history')
def history_page():
    return send_from_directory(app.static_folder, 'history.html')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# ---------- MAIN ----------
if __name__ == '__main__':
    init_db()
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
