# predict.py
import argparse, json, numpy as np, os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

IMG_SIZE = (224,224)
MODEL_PATH = 'models/mobilenetv2_model_updated.h5'            # or mobilenetv2_model_updated.h5
LABELS_PATH = 'models/class_labels.json'
REMEDIES_PATH = 'models/remedies.json'
1
parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='Path to input image')
args = parser.parse_args()
img_path = args.image

# Load model
if not os.path.exists(MODEL_PATH):
    # fallback to updated model
    alt = 'models/mobilenetv2_model_updated.h5'
    if os.path.exists(alt):
        MODEL_PATH = alt
    else:
        raise FileNotFoundError("No saved model found at models/")

model = load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels_map = json.load(f)
# labels_map keys are strings, convert to int->str map
labels = {int(k): v for k, v in labels_map.items()}

# Load remedies
if os.path.exists(REMEDIES_PATH):
    with open(REMEDIES_PATH, 'r') as f:
        remedies = json.load(f)
else:
    remedies = {}

# Prepare image
img = load_img(img_path, target_size=IMG_SIZE)
x = img_to_array(img)
x = mobilenet_preprocess(x)
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)
idx = int(np.argmax(pred))
conf = float(np.max(pred))
pred_class = labels[idx]

print("\nPrediction:", pred_class)
print("Confidence: {:.3f}".format(conf))

# Print remedy
info = remedies.get(pred_class)
if info:
    print("\n=== Suggested Remedy ===")
    if info.get("common_name"):
        print("Disease:", info["common_name"])
    if info.get("remedy"):
        print("Remedy:", info["remedy"])
    if info.get("prevention"):
        print("Prevention:", info["prevention"])
else:
    print("\nNo remedy info available for this class. Add entry to models/remedies.json")
