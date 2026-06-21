import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "."
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 5
MODEL_OUT = "dr_model.keras"
CLASS_INDICES_OUT = "class_indices.json"

TRAIN_DIR = os.path.join(DATA_DIR, "training")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")


def build_generators():
    """Creates train/val/test data generators with augmentation on train only."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
    )
    # No augmentation for val/test — we want a clean evaluation signal
    plain = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = plain.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = plain.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model():
    """MobileNetV2 backbone (frozen) + small trainable classification head."""
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze pretrained weights — fast on CPU

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_weights(train_gen):
    """Diabetic retinopathy datasets are heavily imbalanced (mostly class 0).
    This computes per-class weights so rare classes aren't ignored."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights (to correct imbalance):", class_weight_dict)
    return class_weight_dict


def main():
    print("Loading data...")
    train_gen, val_gen, test_gen = build_generators()

    print("Class index mapping:", train_gen.class_indices)
    with open(CLASS_INDICES_OUT, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    class_weights = compute_weights(train_gen)

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_loss", save_best_only=True
        ),
    ]

    print("Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    model.save(MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved class index mapping to {CLASS_INDICES_OUT}")


if __name__ == "__main__":
    main()
