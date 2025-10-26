import os, json, io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

# Try import tensorflow/keras - provide friendly error if missing
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "model.h5")
CLASSES_JSON = os.path.join(BASE_DIR, "saved_model", "classes.json")

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

model = None
classes = None

def load_resources():
    global model, classes
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available. Install tensorflow before running the backend.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found. Train the model first using train.py and ensure {MODEL_PATH} exists.")
    model = load_model(MODEL_PATH)
    if os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, 'r') as f:
            inv_map = json.load(f)
            # Ensure keys are ints
            classes = {int(k):v for k,v in inv_map.items()}
    else:
        # default mapping (best-effort)
        classes = {0:'fake', 1:'real'}
    print("Model and classes loaded. Classes:", classes)

def preprocess_image(image_bytes, target_size=(128,128)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        try:
            load_resources()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    if 'file' not in request.files:
        return jsonify({"error":"No file sent"}), 400
    f = request.files['file']
    img_bytes = f.read()
    try:
        inp = preprocess_image(img_bytes)
        pred = model.predict(inp)[0][0]  # sigmoid output
        # pred is probability for class '1' as per training generator (class mapping saved)
        prob = float(pred)
        # Determine label using class mapping: we must check which index corresponds to 'real' or 'fake'.
        # Our train script saved mapping inv_map where index->class_name
        # If classes mapping maps 0->'fake' and 1->'real', prob closer to 1 means 'real'.
        # We'll pick the label based on threshold 0.5
        label_idx = 1 if prob >= 0.5 else 0
        label_name = classes.get(label_idx, "real" if label_idx==1 else "fake")
        confidence = prob if label_idx==1 else 1-prob
        explanation = f"Model probability (raw): {prob:.4f}"
        return jsonify({"label": label_name, "confidence": round(float(confidence), 4), "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
