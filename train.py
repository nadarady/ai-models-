import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

TRAIN_DIR = "training"
VAL_DIR = "validation"
TEST_DIR = "test"

IMG_SIZE = 224  # Expanded resolution so model can see micro-anomalies
BATCH_SIZE = 16  # Dropped slightly to accommodate fine-tuning memory footprint
EPOCHS = 20
NUM_CLASSES = 5
MODEL_OUT = "dr_model.keras"


def get_class_weights(generator):
    """Calculates balanced class weights natively using pure numpy to avoid venv dependency errors."""
    classes = generator.classes
    total_samples = len(classes)
    num_classes = len(np.unique(classes))
    class_counts = np.bincount(classes)
    class_weights = total_samples / (num_classes * class_counts)
    return {i: float(weight) for i, weight in enumerate(class_weights)}


def build_generators():
    # Utilizing native MobileNetV2 preprocessing instead of standard 1/255 division
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    plain = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Explicit class lists block dimension mismatches if a subset folder skips a class
    class_list = ["0", "1", "2", "3", "4"]

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        classes=class_list,
    )
    val_gen = plain.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=class_list,
    )
    test_gen = plain.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=class_list,
    )
    return train_gen, val_gen, test_gen


def build_model():
    # Keep weights locked to ImageNet foundations
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )

    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)  # Expanded layer capacity
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Reverted learning rate to 1e-3 since the base is safely frozen
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading data...")
    train_gen, val_gen, test_gen = build_generators()

    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)
    print("Class index mapping saved.")

    class_weights = get_class_weights(train_gen)
    print(f"Class weights (to correct imbalance): {class_weights}")

    print("Building model...")
    model = build_model()
    model.summary()

    # Callbacks manage structural checkpoints
    checkpoint = ModelCheckpoint(
        MODEL_OUT, monitor="val_loss", save_best_only=True, mode="min", verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )

    print("Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    # FIX: Final manual save removed. ModelCheckpoint preserves the peak performance file.
    print(f"\nTraining complete. Premium weights safely retained inside '{MODEL_OUT}'")


if __name__ == "__main__":
    main()
