"""
app.py
Flask web app that serves a simple UI for uploading a retina image
and getting a diabetic retinopathy severity prediction.

Run (after training has produced dr_model.keras + class_indices.json):
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "dr_model.keras")
CLASS_INDICES_PATH = os.path.join(APP_DIR, "class_indices.json")
IMG_SIZE = 128

LABELS = {
    "0": "No DR (Healthy)",
    "1": "Mild DR",
    "2": "Moderate DR",
    "3": "Severe DR",
    "4": "Proliferative DR",
}

app = Flask(__name__)

print("Loading model...")
model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)  # e.g. {"0": 0, "1": 1, ...}
# invert so we can map model output index -> folder/class name
index_to_class = {v: k for k, v in class_indices.items()}
print("Model loaded. Class mapping:", index_to_class)


def predict_image(file_path):
    img = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]  # array of 5 probabilities
    best_idx = int(np.argmax(preds))
    class_name = index_to_class[best_idx]
    confidence = float(preds[best_idx])

    return {
        "class": class_name,
        "label": LABELS.get(class_name, class_name),
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            LABELS.get(index_to_class[i], index_to_class[i]): round(float(p) * 100, 2)
            for i, p in enumerate(preds)
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_path = os.path.join(APP_DIR, "temp_upload.jpg")
    file.save(upload_path)

    try:
        result = predict_image(upload_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
