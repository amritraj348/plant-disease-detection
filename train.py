"""
train.py
Fixed training script for Mac M1: loads/saves labels, fine-tunes safely, and removes multiprocessing args.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_FOLDER = "./plant_dataset"   # <- ensure this exists and contains class subfolders
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2_model.h5")
MODEL_UPDATED_PATH = os.path.join(MODEL_DIR, "mobilenetv2_model_updated.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

# Fine-tune config
FORCE_RETRAIN = False   # If True -> retrain from scratch even if MODEL_PATH exists
UNFREEZE_LAST_N = 30    # number of top layers to unfreeze for fine-tuning
FINETUNE_EPOCHS = 10
TRAIN_EPOCHS = 20       # if you retrain from scratch
LEARNING_RATE_FINETUNE = 1e-5
LEARNING_RATE_SCRATCH = 1e-4

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=mobilenet_preprocess,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    DATASET_FOLDER,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    DATASET_FOLDER,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print(f"\nFound {train_data.samples} training images and {val_data.samples} validation images")
print("Number of classes:", NUM_CLASSES)
print("Class indices:", train_data.class_indices)

# Save/overwrite the class label mapping
with open(LABELS_PATH, "w") as f:
    json.dump({v: k for k, v in train_data.class_indices.items()}, f, indent=2)
print("Saved label mapping to:", LABELS_PATH)

# -----------------------------
# BUILD MODEL FUNCTION
# -----------------------------
def build_model(num_classes):
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # Freeze most layers initially
    for layer in base.layers[:-40]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base.input, outputs=out)
    return model

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, "mobilenetv2_best.h5"), save_best_only=True, monitor="val_loss", verbose=1)

# -----------------------------
# LOAD OR TRAIN (flexible mode)
# -----------------------------
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "models/mobilenetv2_model_updated.h5"
LABELS_PATH = "models/class_labels.json"

def load_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")
    with open(path, "r") as f:
        labels_map = json.load(f)

    # Normalize to index → name dictionary
    index_to_name = {}
    try:
        for k, v in labels_map.items():  # case: {"0":"Apple..."}
            index_to_name[int(k)] = v
    except:
        for name, idx in labels_map.items():  # case: {"Apple...":0}
            index_to_name[int(idx)] = name

    return index_to_name


def main():
    print("🔍 Checking for saved fine-tuned model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Saved model not found at {MODEL_PATH}\n"
            "Make sure mobilenetv2_model_updated.h5 exists in models/"
        )

    print(f"📌 Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)

    print("✅ Model loaded successfully!")

    # Load class labels
    labels = load_labels(LABELS_PATH)
    print(f"📄 Loaded {len(labels)} classes:")
    for idx, name in labels.items():
        print(f"{idx}: {name}")

    print("\n✨ Your model is ready for prediction!")


if __name__ == "__main__":
    main()




# -----------------------------
# PLOT TRAINING CURVES (if available)
# -----------------------------
if 'history' in locals() and history is not None:
    try:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history.get("accuracy", []), label="train")
        plt.plot(history.history.get("val_accuracy", []), label="val")
        plt.legend(); plt.title("Accuracy")
        plt.subplot(1,2,2)
        plt.plot(history.history.get("loss", []), label="train")
        plt.plot(history.history.get("val_loss", []), label="val")
        plt.legend(); plt.title("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
        print("Saved training curves to:", os.path.join(MODEL_DIR, "training_curves.png"))
        plt.show()
    except Exception as e:
        print("Could not plot training curves:", e)

# -----------------------------
# EVALUATION: Confusion Matrix & Classification Report
# -----------------------------
print("\n🔎 Evaluating on validation set...")

# Ensure `model` exists. If not, try to load the saved model.
try:
    model  # check if model variable exists
except NameError:
    # Try to load the saved model from disk
    fallback_paths = [
        os.path.join(MODEL_DIR, "mobilenetv2_model_updated.h5"),
        os.path.join(MODEL_DIR, "mobilenetv2_model.h5"),
        os.path.join(MODEL_DIR, "mobilenetv2_best.h5"),
        os.path.join(MODEL_DIR, "mobilenetv2_best.hdf5"),
    ]
    loaded = False
    for p in fallback_paths:
        if os.path.exists(p):
            print(f"Loading model from: {p}")
            model = tf.keras.models.load_model(p)
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(
            "No model found in memory and no fallback model file exists in models/. "
            "Train the model or ensure MODEL_PATH points to a valid saved model."
        )

# Now predict on validation generator
# Note: val_data must be defined (ImageDataGenerator.flow_from_directory with subset='validation')
preds = model.predict(val_data, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(train_data.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print("Saved confusion matrix to:", cm_path)
plt.show()

print("\n✅ Done.")
