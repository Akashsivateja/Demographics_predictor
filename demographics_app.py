# app.py
import streamlit as st
import tensorflow as tf  # Corrected import
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Ensure you use the correct preprocessing function matching your training
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
import traceback # Added for detailed error logging if needed

# --- Configuration ---
# Match these with your Colab training settings (Cell 6/7)
IMG_SIZE = (224, 224)
# --- IMPORTANT: Use the MULTI-OUTPUT model filename ---
# Make sure this exact file exists in the same directory as app.py
MODEL_FILENAME = "multi_output_demographics_model_best.h5"

# Define the expected outputs and their types (must match model heads)
EXPECTED_OUTPUTS = {
    'gender': 'classification',
    'age': 'regression',
    'height': 'regression',
    'weight': 'regression'
}
# Class names for gender output (must match LabelEncoder from Colab Cell 6)
CLASS_NAMES_GENDER = ['female', 'male'] # Adjust if your encoding was different

# --- Model Loading ---
# Cache the loaded model to avoid reloading on every interaction
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model from the specified path."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure it's in the same directory as app.py.")
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        # Load the multi-output model
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        # Optional: Verify expected output names match loaded model
        loaded_output_names = list(model.output_names)
        expected_names = list(EXPECTED_OUTPUTS.keys())
        if set(loaded_output_names) != set(expected_names):
            st.warning(f"Model output name mismatch! Expected: {expected_names}, Found in model: {loaded_output_names}")
            print(f"WARNING: Model output name mismatch! Expected: {expected_names}, Found in model: {loaded_output_names}")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model from {model_path}: {e}")
        print(f"Error loading model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_image_streamlit(img_pil, target_size):
    """Prepares a PIL image for the ResNet50 multi-output model."""
    try:
        # Convert PIL image to NumPy array
        img_array = image.img_to_array(img_pil)

        # Ensure 3 channels (handle grayscale/RGBA automatically if possible)
        if img_array.ndim == 2: # Grayscale
             img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 1: # Grayscale with channel dim
             img_array = np.concatenate([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4: # RGBA
             img_array = img_array[..., :3] # Take only RGB channels
        # Add check if still not 3 channels
        if img_array.shape[-1] != 3:
             st.error(f"Image preprocessing failed: Unexpected number of channels ({img_array.shape[-1]}). Expected 3 (RGB).")
             return None

        # Resize (using TF for consistency, ensure float32)
        img_tf = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_resized = tf.image.resize(img_tf, target_size) # target_size is (height, width)
        # Add batch dimension -> (1, height, width, channels)
        img_expanded = tf.expand_dims(img_resized, axis=0)
        # Preprocess using the specific function for ResNet50
        img_preprocessed = preprocess_input(img_expanded) # Applies scaling etc.
        return img_preprocessed # Returns a TensorFlow tensor ready for model input
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        print(f"Error preprocessing image: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Demographics Predictor")
st.title("👤 Image Demographics Predictor (Multi-Output)")
st.write("Upload an image to predict Gender, Age, Height, and Weight using a trained model.")
st.caption(f"Model input size: {IMG_SIZE}x{IMG_SIZE} pixels.")
st.divider()

# Load the model (will be cached after first run)
# Place the model loading attempt early, so errors are shown quickly
model = load_keras_model(MODEL_FILENAME)

# Sidebar Information
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a deep learning model (based on ResNet50) "
    "trained to predict multiple demographic attributes from facial images. "
    "Upload an image to see the predictions."
)
st.sidebar.header("Model Info")
if model is not None:
    st.sidebar.markdown(f"**Model File:** `{MODEL_FILENAME}`")
    st.sidebar.markdown(f"**Input Size:** `{IMG_SIZE}`")
    st.sidebar.markdown("**Outputs:**")
    for name, type in EXPECTED_OUTPUTS.items():
        # Check if the output actually exists in the loaded model before listing
        if name in model.output_names:
             st.sidebar.markdown(f"- {name.capitalize()} ({type})")
        else:
             st.sidebar.markdown(f"- ~~{name.capitalize()} ({type})~~ (Not in loaded model)")
else:
     st.sidebar.error("Model failed to load.")

st.sidebar.warning("Disclaimer: Predictions are estimates based on the training data and may not be fully accurate, especially for height and weight derived solely from images.")


# Main area for upload and results
col1, col2 = st.columns([2, 3]) # Adjust ratio as needed

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Display uploaded image safely
            st.image(uploaded_file, caption="Uploaded Image", use_column_width='always')
        except Exception as e:
             st.error(f"Error displaying uploaded image: {e}")


with col2:
    st.subheader("Prediction Results")
    results_placeholder = st.empty() # Placeholder to show results or 'waiting' message

    if uploaded_file is not None:
        if model is not None:
            # Only proceed if model is loaded successfully
            with st.spinner('Analyzing image and predicting...'):
                try:
                    # Open image using PIL, ensure RGB
                    img_pil = Image.open(uploaded_file).convert('RGB')

                    # Preprocess the image (returns TF Tensor)
                    processed_img_tensor = preprocess_image_streamlit(img_pil, IMG_SIZE)

                    if processed_img_tensor is not None:
                        # Make prediction - returns a dictionary for multi-output models
                        prediction_dict = model.predict(processed_img_tensor)

                        # --- Display Predictions ---
                        results_container = results_placeholder.container() # Use the placeholder

                        # Iterate through expected outputs and display formatted predictions
                        for output_name, task_type in EXPECTED_OUTPUTS.items():
                            if output_name not in prediction_dict:
                                results_container.warning(f"Output '{output_name}' not found in model prediction dictionary.")
                                continue

                            # Prediction value is usually [[value]] for single output heads
                            # Add checks for potential issues in prediction output format
                            if isinstance(prediction_dict[output_name], np.ndarray) and prediction_dict[output_name].size > 0:
                                 pred_value_raw = prediction_dict[output_name].flatten()[0] # Safely get first element
                            else:
                                 results_container.error(f"Unexpected prediction format for '{output_name}'.")
                                 continue


                            # Format based on task type
                            if task_type == 'classification' and output_name == 'gender':
                                pred_prob = pred_value_raw # Probability of class 1 ('male')
                                threshold = 0.5
                                pred_class_index = int(pred_prob > threshold)
                                pred_class_name = CLASS_NAMES_GENDER[pred_class_index] if pred_class_index < len(CLASS_NAMES_GENDER) else f"Class {pred_class_index}"
                                confidence = pred_prob if pred_class_index == 1 else 1 - pred_prob

                                results_container.metric(label=f"Predicted Gender", value=pred_class_name.capitalize())
                                results_container.progress(confidence, text=f"{confidence:.1%} confidence")

                            elif task_type == 'regression':
                                pred_value = pred_value_raw
                                unit = ""
                                label_text = f"Predicted {output_name.capitalize()}"
                                if output_name == 'age': unit = "years"
                                elif output_name == 'height': unit = "inches"
                                elif output_name == 'weight': unit = "kg"
                                results_container.metric(label=label_text, value=f"{pred_value:.1f} {unit}")

                            else:
                                results_container.warning(f"Unsupported task type '{task_type}' for output '{output_name}'.")
                        st.divider() # Add a visual separator after results

                    else:
                        # Error message already shown by preprocess_image_streamlit
                        # results_placeholder.error("Image could not be preprocessed.")
                        pass # Avoid duplicating error messages

                except Exception as e:
                    results_placeholder.error(f"An critical error occurred during prediction: {e}")
                    print(f"Prediction loop error: {e}")
                    traceback.print_exc() # Print detailed error to console/logs
        else:
            results_placeholder.error("Model is not loaded. Cannot make predictions. Check model file path and loading logs.")
    else:
        results_placeholder.info("Upload an image using the button on the left to see the predictions.")
