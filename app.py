from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "waste_classifier.keras"
CLASS_PATH = APP_DIR / "models" / "class_names.json"
HISTORY_PATH = APP_DIR / "outputs" / "predictions.csv"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.25rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {font-size: 1rem; color: #64748b; margin-bottom: 1.3rem;}
    .result-card {padding: 1.15rem; border: 1px solid rgba(128,128,128,.25); border-radius: 14px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">♻️ Smart Waste Classification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Classify common waste items from a webcam photo or uploaded image using a trained image-classification model.</div>',
    unsafe_allow_html=True,
)

WASTE_GUIDANCE = {
    "cardboard": "Keep cardboard clean and dry where possible. Check local recycling rules before disposal.",
    "glass": "Handle broken glass carefully. Follow your local glass collection and safety rules.",
    "metal": "Empty containers before recycling and check local acceptance rules.",
    "paper": "Keep paper dry and free from food contamination where possible.",
    "plastic": "Check the resin type and local recycling rules; not every plastic item is accepted everywhere.",
    "trash": "The model classifies this as general trash. Check local waste guidance for safe disposal.",
}


def read_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "category", "confidence_percent", "source"])
    try:
        return pd.read_csv(HISTORY_PATH)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=["timestamp", "category", "confidence_percent", "source"])


def save_prediction(category: str, confidence: float, source: str) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_PATH.exists() and HISTORY_PATH.stat().st_size > 0
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["timestamp", "category", "confidence_percent", "source"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            category,
            round(confidence * 100, 2),
            source,
        ])


@st.cache_resource(show_spinner="Loading the trained waste classifier...")
def load_model_and_classes():
    # TensorFlow is imported only when the app actually has a trained model to load.
    import tensorflow as tf

    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with CLASS_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    return model, class_names


def prepare_image(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(rgb, dtype=np.float32), axis=0)


def get_prediction(model, class_names: list[str], image: Image.Image):
    probabilities = model.predict(prepare_image(image), verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
    top_predictions = [(class_names[int(i)], float(probabilities[int(i)])) for i in top_indices]
    return class_names[best_index], float(probabilities[best_index]), top_predictions


with st.sidebar:
    st.header("How to use")
    st.markdown("1. Take a webcam photo or upload a waste image.\n2. Click **Classify waste**.\n3. Review the category and confidence.\n4. Review past predictions under **History & analytics**.")
    st.divider()
    st.caption("This is a student prototype. Predictions may be wrong; confirm sorting decisions using local waste guidance.")

input_tab, history_tab, about_tab = st.tabs(["🔎 Classify waste", "📊 History & analytics", "ℹ️ About project"])

with input_tab:
    source_choice = st.radio("Choose image source", ["Use webcam", "Upload image"], horizontal=True)
    image = None
    source_label = "webcam"

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        if source_choice == "Use webcam":
            captured = st.camera_input("Take a photo of one waste item", key="waste_camera")
            if captured is not None:
                try:
                    image = Image.open(captured).convert("RGB")
                    source_label = "webcam"
                except Exception as exc:
                    st.error(f"Could not read the camera image: {exc}")
        else:
            upload = st.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
            if upload is not None:
                try:
                    image = Image.open(upload).convert("RGB")
                    source_label = "upload"
                except Exception as exc:
                    st.error(f"Could not read the uploaded image: {exc}")

        if image is not None:
            st.image(image, caption="Image to classify", use_container_width=True)

    with right:
        st.subheader("Prediction")
        if image is None:
            st.info("Take a photo with your webcam or upload an image to begin.")
        else:
            st.write("Tip: place one item near the center, use good lighting, and keep the background simple.")
            if st.button("Classify waste", type="primary", use_container_width=True):
                try:
                    model, class_names = load_model_and_classes()
                    if model is None:
                        st.error("The trained model was not found. Follow the setup guide in README.md, place the dataset in the dataset folder, and run train_model.py first.")
                    else:
                        category, confidence, top_predictions = get_prediction(model, class_names, image)
                        st.session_state["last_result"] = {
                            "category": category,
                            "confidence": confidence,
                            "top_predictions": top_predictions,
                            "image": image,
                            "source": source_label,
                        }
                        save_prediction(category, confidence, source_label)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        result = st.session_state.get("last_result")
        if result is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.caption("Predicted category")
            st.subheader(result["category"].replace("_", " ").title())
            st.metric("Model confidence", f"{result['confidence'] * 100:.1f}%")
            st.progress(min(max(float(result["confidence"]), 0.0), 1.0))
            if result["confidence"] < 0.55:
                st.warning("Low confidence. Try a clearer image or a different angle; do not rely on this prediction alone.")
            st.write(WASTE_GUIDANCE.get(result["category"].lower(), "Check local waste disposal guidance."))
            st.markdown("**Top predictions**")
            for label, probability in result["top_predictions"]:
                st.write(f"{label.title()} — {probability * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

with history_tab:
    history = read_history()
    if history.empty:
        st.info("No predictions saved yet. Classify an item first to build your history.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", len(history))
        c2.metric("Categories seen", history["category"].nunique())
        c3.metric("Average confidence", f"{pd.to_numeric(history['confidence_percent'], errors='coerce').mean():.1f}%")
        st.subheader("Predictions by category")
        category_counts = history["category"].value_counts().rename_axis("Category").to_frame("Predictions")
        st.bar_chart(category_counts)
        st.subheader("Recent prediction history")
        st.dataframe(history.tail(100).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download history as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="waste_prediction_history.csv",
            mime="text/csv",
        )

with about_tab:
    st.subheader("Project objective")
    st.write("Classify a single waste item into one of the trained categories and record the prediction for simple analytics.")
    st.subheader("Model approach")
    st.write("The included training script uses transfer learning with MobileNetV2, a pretrained image model, and fine-tunes a small classification head for the waste categories in your dataset.")
    st.subheader("Supported categories")
    st.write("The TrashNet-style dataset uses cardboard, glass, metal, paper, plastic, and trash. The available categories are determined by the class folders used during training.")
    st.warning("The app classifies images; it does not physically sort waste, identify every material type, or determine local recycling eligibility with certainty.")
