import argparse, os, sys, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to image file")
parser.add_argument("--model", default="backend/saved_model/model.h5", help="Path to model file")
parser.add_argument("--classes", default="backend/saved_model/classes.json", help="Path to classes json")
args = parser.parse_args()

if not os.path.exists(args.model):
    print("Model not found:", args.model)
    sys.exit(1)

m = load_model(args.model)
with open(args.classes, 'r') as f:
    inv_map = json.load(f)
    classes = {int(k):v for k,v in inv_map.items()}

img = image.load_img(args.image, target_size=(128,128))
arr = image.img_to_array(img)/255.0
arr = np.expand_dims(arr, 0)
pred = m.predict(arr)[0][0]
label_idx = 1 if pred>=0.5 else 0
label_name = classes.get(label_idx, "real" if label_idx==1 else "fake")
confidence = pred if label_idx==1 else 1-pred
print(f"Prediction: {label_name} ({confidence*100:.2f}%)")