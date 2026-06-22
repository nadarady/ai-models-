import tensorflow as tf

# Load your current 9MB model
model = tf.keras.models.load_model("dr_model.keras")

# Convert it to the lightweight TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the new ultra-optimized file
with open("dr_model.tflite", "wb") as f:
    f.write(tflite_model)
