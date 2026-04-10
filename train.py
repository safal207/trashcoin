"""
Training script for TrashNet waste classifier using MobileNetV2 transfer learning.

Usage:
    python train.py [--epochs 10] [--output model/trashnet_model.keras]

The script downloads the TrashNet dataset via tensorflow-datasets and trains a
MobileNetV2-based classifier. Trained weights are saved to the output path.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

try:
    import tensorflow_datasets as tfds
except ImportError:
    raise SystemExit(
        "tensorflow-datasets is required. Run: pip install tensorflow-datasets"
    )

# TrashNet class order as returned by tensorflow_datasets (alphabetical)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 224
BATCH_SIZE = 32


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def preprocess(sample):
    img = tf.cast(sample["image"], tf.float32) / 255.0
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    label = sample["label"]
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, label


def load_datasets():
    ds_train, ds_val = tfds.load(
        "trashnet",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=False,
    )

    train = (
        ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1024)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val = (
        ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train, val


def fine_tune(model: tf.keras.Model, train_ds, val_ds, epochs: int):
    """Unfreeze last 30 layers of MobileNetV2 and fine-tune."""
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)


def train(epochs: int, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("Loading TrashNet dataset...")
    train_ds, val_ds = load_datasets()

    print("Building model...")
    model = build_model(num_classes=len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(output_path, save_best_only=True, monitor="val_accuracy"),
    ]

    print(f"Training for up to {epochs} epochs (head only)...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    print("Fine-tuning top layers of MobileNetV2...")
    fine_tune(model, train_ds, val_ds, epochs=max(5, epochs // 2))

    model.save(output_path)
    print(f"Model saved to {output_path}")

    # Save class names alongside the model
    classes_path = os.path.join(os.path.dirname(output_path) or ".", "classes.json")
    with open(classes_path, "w") as f:
        json.dump(CLASSES, f)
    print(f"Class names saved to {classes_path}")

    # Quick evaluation
    loss, acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {acc:.4f} | loss: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrashNet classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument(
        "--output",
        default="model/trashnet_model.keras",
        help="Output model path",
    )
    args = parser.parse_args()
    train(args.epochs, args.output)
