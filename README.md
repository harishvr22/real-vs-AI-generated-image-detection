  # Deepfake vs Real Image Detection - Fullstack Project

This project includes a **frontend** (static HTML/CSS/JS) and a **backend** (Flask + Keras) to detect whether an uploaded image is real or AI-generated.

## Structure
```
deepfake_project/
  frontend/            # your UI (index.html, index.js, style.css)
  backend/
    train.py           # train model on dataset/ (will save to backend/saved_model/model.h5)
    app.py             # Flask API - serve frontend and /api/predict endpoint
    predict_cli.py     # simple CLI to test model locally
    saved_model/       # (produced after training) model.h5 and classes.json
    requirements.txt
  dataset/             # put your dataset here (folders: real/, fake/)
```

## Quick start (recommended: use a short path like C:\\tf_project or D:\\tf_env to avoid Windows long-path issues)

1. Create & activate a Python virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# Linux / macOS
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r backend/requirements.txt
```

3. Prepare your dataset:
Place your dataset in the `dataset/` folder with the following structure:
```
dataset/
  real/
    img1.jpg
    ...
  fake/
    img1.jpg
    ...
```

4. Train the model (this will save the best model to `backend/saved_model/model.h5`):
```bash
python backend/train.py
```

5. Run the backend server:
```bash
python backend/app.py
```

6. Open the frontend:
Visit `http://127.0.0.1:5000/` in your browser, upload an image and click Predict.

## Notes and tips
- If TensorFlow installation on Windows fails due to long path errors, either enable long paths in Windows or use Google Colab (recommended) to train the model and then copy `model.h5` to `backend/saved_model/`.
- For faster inference/training you can switch to a GPU environment (Colab or local CUDA-enabled setup).
- The frontend expects the endpoint `/api/predict`. If you host the backend elsewhere, update `PREDICT_ENDPOINT` in `frontend/index.js`.
