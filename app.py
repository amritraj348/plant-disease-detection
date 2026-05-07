# app.py - Streamlit UI for Plant Disease Detection (MobileNetV2 fine-tuned)
import os
# ------------------- quiet TF logs early -------------------
# Set before importing tensorflow where possible
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error

import io
import logging
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Keras/TensorFlow related imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ------------------- Logging -------------------
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ------------------- Config -------------------
MODEL_PATH_DEFAULT = "models/mobilenetv2_model_updated.h5"
FALLBACK_MODEL = "models/mobilenetv2_model.h5"
LABELS_PATH = "models/class_labels.json"
REMEDIES_PATH = "models/remedies.json"
IMG_SIZE = (224, 224)

# ------------------- Helpers -------------------
def find_model_path(preferred=MODEL_PATH_DEFAULT):
    if preferred and Path(preferred).exists():
        return preferred
    if Path(FALLBACK_MODEL).exists():
        return FALLBACK_MODEL
    # try any .h5 in models
    model_dir = Path("models")
    if model_dir.exists():
        h5s = list(model_dir.glob("*.h5"))
        if h5s:
            return str(h5s[0])
    return None

@st.cache_resource(show_spinner=False)
def load_tf_model(path):
    """Load model once and cache it for the Streamlit session."""
    model = load_model(path, compile=False)
    return model

@st.cache_data(show_spinner=False)
def load_labels(path):
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        labels_map = json.load(f)
    # normalize to index->name
    idx2name = {}
    try:
        for k, v in labels_map.items():
            idx2name[int(k)] = v
    except Exception:
        for name, idx in labels_map.items():
            idx2name[int(idx)] = name
    return idx2name

@st.cache_data(show_spinner=False)
def load_remedies(path):
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def preprocess_pil_image(pil_img, target_size=IMG_SIZE):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = img_to_array(pil_img)
    arr = mobilenet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def pil_from_session_state(sample_dir: Path):
    """
    Reconstruct a PIL.Image from session state:
    - if uploaded_bytes present -> use that
    - elif last_sample present -> load from disk
    Returns PIL.Image or None.
    """
    uploaded_bytes = st.session_state.get("uploaded_bytes")
    last_sample = st.session_state.get("last_sample")
    if uploaded_bytes:
        try:
            return Image.open(io.BytesIO(uploaded_bytes))
        except Exception as e:
            # corrupt upload; clear and return None
            st.session_state["uploaded_bytes"] = None
            st.error("Cannot open uploaded file from session state.")
            st.exception(e)
            return None
    if last_sample:
        sample_path = sample_dir / last_sample
        if sample_path.exists():
            try:
                return Image.open(sample_path)
            except Exception as e:
                st.session_state["last_sample"] = None
                st.error("Cannot open chosen sample image from disk.")
                st.exception(e)
                return None
    return None

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="Plant Disease Detector", layout="wide")
st.title("🌱 Plant Disease Detector (MobileNetV2)")

# left and right columns
col1, col2 = st.columns([1, 2])

# Ensure session_state keys exist (idempotent)
if "uploaded_bytes" not in st.session_state:
    st.session_state["uploaded_bytes"] = None
if "last_sample" not in st.session_state:
    st.session_state["last_sample"] = None
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "prediction_info" not in st.session_state:
    st.session_state["prediction_info"] = {}

with col1:
    st.header("Model")
    model_path = find_model_path()
    if model_path is None:
        st.error("No model found in `models/`. Place `mobilenetv2_model_updated.h5` in the `models/` folder.")
        st.stop()

    st.write("Using model:", f"`{model_path}`")
    try:
        model = load_tf_model(model_path)
        st.success("Model loaded ✅")
    except Exception as e:
        st.error("Failed to load model. See details below.")
        st.exception(e)
        st.stop()

    labels = load_labels(LABELS_PATH)
    if labels:
        st.write(f"Classes loaded: {len(labels)}")
    else:
        st.warning("Labels file not found or empty (`models/class_labels.json`). Predictions will show indices.")

    remedies = load_remedies(REMEDIES_PATH)
    if remedies:
        st.write(f"Remedies loaded: {len(remedies)} entries")
    else:
        st.info("No remedies.json found in models/. You can add one to show suggested remedies.")

    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("- Upload an image (left) or pick a sample image (below).")
    st.markdown("- App uses fine-tuned `mobilenetv2_model_updated.h5` by default.")
    st.markdown("- Run: `streamlit run app.py`")

with col2:
    st.header("Predict")

    # sample images list
    sample_dir = Path("test_images")
    sample_images = []
    if sample_dir.exists():
        # include common image extensions
        sample_images = sorted([p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    # sample picker
    sample_choice = None
    if sample_images:
        sample_names = ["-- none --"] + [p.name for p in sample_images]
        sample_choice = st.selectbox("Or choose a sample image", sample_names)
    else:
        st.info("Place sample images (jpg/png) in `test_images/` folder to use sample picker.")

    # file uploader
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

    # If user uploaded a file this run, save bytes into session_state
    if uploaded is not None:
        try:
            st.session_state["uploaded_bytes"] = uploaded.getvalue()
            # clear previous sample selection
            st.session_state["last_sample"] = None
            # clear previous prediction when a new file arrives
            st.session_state["prediction"] = None
            st.session_state["prediction_info"] = {}
        except Exception as e:
            st.error("Failed to read uploaded file.")
            st.exception(e)

    # If user selected a sample this run, persist the selection
    if sample_choice and sample_choice != "-- none --":
        st.session_state["last_sample"] = sample_choice
        # clear uploaded bytes when user explicitly chooses a sample
        st.session_state["uploaded_bytes"] = None
        # clear previous prediction when a new sample chosen
        st.session_state["prediction"] = None
        st.session_state["prediction_info"] = {}

    # Reconstruct PIL image from session_state or disk
    pil_image = pil_from_session_state(sample_dir)

    if pil_image is not None:
        st.image(pil_image, caption="Input image", use_column_width=True)

        # Use a form so the predict action is explicit and survives reruns
        with st.form(key="predict_form"):
            submitted = st.form_submit_button("Predict")
            if submitted:
                try:
                    x = preprocess_pil_image(pil_image)
                    preds = model.predict(x, verbose=0)
                    idx = int(np.argmax(preds))
                    conf = float(np.max(preds))
                    pred_name = labels.get(idx, str(idx))

                    # store prediction in session state so it persists on reconnect/tab switch
                    st.session_state["prediction"] = pred_name
                    st.session_state["prediction_info"] = {"conf": conf, "idx": idx}

                except Exception as e:
                    st.error("Prediction failed.")
                    st.exception(e)

        # If we already have a stored prediction, show it (this survives tab switches)
        if st.session_state.get("prediction"):
            pred_name = st.session_state["prediction"]
            conf = st.session_state["prediction_info"].get("conf", 0.0)
            st.subheader("Prediction")
            st.write(f"**{pred_name}**  —  Confidence: {conf:.4f}")

            # show remedy if available
            rem = remedies.get(pred_name)
            if rem:
                st.markdown("#### Suggested remedy / notes")
                if isinstance(rem, str):
                    st.write(rem)
                elif isinstance(rem, dict):
                    if rem.get("common_name"):
                        st.write(f"**Disease:** {rem.get('common_name')}")
                    if rem.get("remedy"):
                        st.write(f"**Remedy:** {rem.get('remedy')}")
                    if rem.get("prevention"):
                        st.write(f"**Prevention:** {rem.get('prevention')}")
            else:
                st.info("No remedy entry found for this class. Add it to `models/remedies.json`.")
    else:
        st.write("Upload or select an image to start prediction.")

# ------------------- Footer -------------------
st.markdown("---")
st.caption(
    "Tip: If TensorFlow prints warnings about 'compiled metrics' or 'missing ScriptRunContext' you can ignore them. Run app with `streamlit run app.py`."
)
