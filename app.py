import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "dr_model.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = 224

# Global initialization
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print(f"Warning: {MODEL_PATH} missing. Complete a training run first.")

if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r") as f:
        raw_indices = json.load(f)
        labels_map = {v: k for k, v in raw_indices.items()}
else:
    labels_map = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}


def predict_image(file_stream):
    if model is None:
        return {"error": "Target neural network model context uninitialized."}

    # Open byte stream directly to bypass operating system file locks
    img = Image.open(file_stream).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = img_to_array(img)
    arr = preprocess_input(arr)  # Pipeline matches structural training logic
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr)[0]
    predicted_class_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class_idx])
    predicted_label = labels_map.get(predicted_class_idx, str(predicted_class_idx))

    # Detailed matrix breakdown for frontend UI rendering
    breakdown = {
        labels_map.get(i, str(i)): float(prob) for i, prob in enumerate(predictions)
    }

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "breakdown": breakdown,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image payload found in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Blank filename received"}), 400

    try:
        img_stream = io.BytesIO(file.read())
        result = predict_image(img_stream)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # use_reloader=False stops double loading conflicts and thread locks
    app.run(debug=True, port=5000, use_reloader=False)
