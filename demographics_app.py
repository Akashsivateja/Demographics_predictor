# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
import traceback

# --- Configuration ---
IMG_SIZE = (224, 224)
MODEL_FILENAME = "multi_output_demographics_model_best.h5" # Ensure this file is present
EXPECTED_OUTPUTS = {
    'gender': 'classification',
    'age': 'regression',
    'height': 'regression',
    'weight': 'regression'
}
CLASS_NAMES_GENDER = ['female', 'male']

# --- Model Loading ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: '{model_path}'")
        return None
    try:
        model = load_model(model_path, compile=False) # Add compile=False for faster loading if not retraining
        print(f"Model loaded: {model_path}")
        # Optional: Verification (can be commented out after first successful run)
        # loaded_output_names = list(model.output_names)
        # expected_names = list(EXPECTED_OUTPUTS.keys())
        # if set(loaded_output_names) != set(expected_names):
        #     st.warning(f"Model output name mismatch! Expected: {expected_names}, Found: {loaded_output_names}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image_streamlit(img_pil, target_size):
    """Prepares a PIL image for the model."""
    try:
        img_array = image.img_to_array(img_pil)
        if img_array.ndim == 2: img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 1: img_array = np.concatenate([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4: img_array = img_array[..., :3]
        if img_array.shape[-1] != 3:
             st.error(f"Preprocessing failed: Unexpected channels ({img_array.shape[-1]}).")
             return None
        img_tf = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_resized = tf.image.resize(img_tf, target_size)
        img_expanded = tf.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_expanded)
        return img_preprocessed
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Demographics Predictor")
st.title("👤 Image Demographics Predictor (Multi-Output)")
st.write("Upload an image to predict Gender, Age, Height, and Weight.")
st.caption(f"Model input size: {IMG_SIZE}x{IMG_SIZE} pixels.")
st.divider()

# Load model early
model = load_keras_model(MODEL_FILENAME)

# Sidebar
st.sidebar.header("About")
st.sidebar.info("App predicts demographic attributes using a ResNet50-based multi-output model.")
st.sidebar.header("Model Info")
if model is not None:
    st.sidebar.markdown(f"**File:** `{MODEL_FILENAME}`")
    st.sidebar.markdown(f"**Input:** `{IMG_SIZE}`")
    st.sidebar.markdown("**Outputs:**")
    for name in EXPECTED_OUTPUTS:
        if name in model.output_names: st.sidebar.markdown(f"- {name.capitalize()}")
        else: st.sidebar.markdown(f"- ~~{name.capitalize()}~~ (Not in model)")
else:
     st.sidebar.error("Model not loaded.")
st.sidebar.warning("Disclaimer: Predictions are estimates. Height/weight accuracy may be limited.")

# Main area
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose image (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # FIX 1: Use use_container_width instead of use_column_width
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
             st.error(f"Error displaying image: {e}")

with col2:
    st.subheader("Prediction Results")
    results_placeholder = st.empty()

    if uploaded_file is not None:
        if model is not None:
            with st.spinner('Predicting...'):
                try:
                    img_pil = Image.open(uploaded_file).convert('RGB')
                    processed_img_tensor = preprocess_image_streamlit(img_pil, IMG_SIZE)

                    if processed_img_tensor is not None:
                        prediction_dict = model.predict(processed_img_tensor)
                        results_container = results_placeholder.container()

                        for output_name, task_type in EXPECTED_OUTPUTS.items():
                            if output_name not in prediction_dict:
                                results_container.warning(f"Output '{output_name}' missing.")
                                continue

                            if isinstance(prediction_dict[output_name], np.ndarray) and prediction_dict[output_name].size > 0:
                                 pred_value_raw = prediction_dict[output_name].flatten()[0]
                            else:
                                 results_container.error(f"Bad prediction format for '{output_name}'.")
                                 continue

                            if task_type == 'classification' and output_name == 'gender':
                                pred_prob = float(pred_value_raw) # Ensure standard float
                                threshold = 0.5
                                pred_class_index = int(pred_prob > threshold)
                                pred_class_name = CLASS_NAMES_GENDER[pred_class_index] if pred_class_index < len(CLASS_NAMES_GENDER) else f"Class {pred_class_index}"
                                confidence = float(pred_prob if pred_class_index == 1 else 1 - pred_prob) # Ensure standard float

                                results_container.metric(label="Predicted Gender", value=pred_class_name.capitalize())
                                # FIX 2: Cast confidence to float for st.progress
                                results_container.progress(float(confidence), text=f"{confidence:.1%} confidence")

                            elif task_type == 'regression':
                                pred_value = float(pred_value_raw) # Ensure standard float
                                unit, label_text = "", f"Predicted {output_name.capitalize()}"
                                if output_name == 'age': unit, label_text = "years", "Predicted Age"
                                elif output_name == 'height': unit, label_text = "inches", "Predicted Height"
                                elif output_name == 'weight': unit, label_text = "kg", "Predicted Weight"
                                results_container.metric(label=label_text, value=f"{pred_value:.1f} {unit}")
                        st.divider()
                    # else: Error handled in preprocessing

                except Exception as e:
                    results_placeholder.error(f"Prediction error: {e}")
                    print(f"Prediction loop error: {e}")
                    traceback.print_exc()
        else:
            results_placeholder.error("Model not loaded.")
    else:
        results_placeholder.info("Upload an image.")
