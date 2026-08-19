import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Animal Classification Panel",
    page_icon="🐾",
    layout="wide"
)

# --- LOAD THE YOLO MODEL ---
@st.cache_resource
def load_yolo_model():
    # Point this to your own trained best.pt file
    model_path = 'runs/classify/train-11/weights/best.pt'
    model = YOLO(model_path)
    return model

# Load the model
try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.info("Please make sure the 'runs/classify/train/weights/best.pt' file exists at the correct location.")

# --- UI ---
st.title("🐾 Animal Classification Control Panel")
st.write("Upload a photo and let your YOLOv8 model classify it live.")

st.divider()

# Image upload area
uploaded_file = st.file_uploader(
    "Select or drag and drop an image...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # 1. Open the uploaded image
    image = Image.open(uploaded_file)

    # 2. Split the screen into left and right columns
    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT COLUMN: show the photo
    with col1:
        st.subheader("📷 Uploaded Photo")
        st.image(image, use_container_width=True)

    # RIGHT COLUMN: show the prediction results
    with col2:
        st.subheader("YOLOv8 Model Prediction")

        with st.spinner("Analyzing image..."):
            # The Ultralytics YOLO model accepts a PIL image directly
            results = model(image)

            # Get the data from the first result
            result = results[0]
            probs = result.probs

            # Highest-confidence class and its score
            top1_id = probs.top1
            top1_score = probs.top1conf.item()
            predicted_class = result.names[top1_id]

            # Probabilities and names for all classes
            all_scores = probs.data.tolist()  # convert tensor to list
            class_names = result.names        # dict structure: {0: 'cat', 1: 'dog', ...}

        # Main prediction cards
        st.success(f"**Predicted Animal:** {predicted_class.upper()}")
        st.metric(label="Confidence", value=f"{top1_score * 100:.2f}%")

        st.divider()
        st.write("**Probability Distribution Across All Classes:**")

        # Sort classes from highest to lowest probability
        sorted_indices = np.argsort(all_scores)[::-1]

        for idx in sorted_indices:
            c_name = class_names[idx]
            c_prob = all_scores[idx]
            percentage = c_prob * 100

            # Highlight: bold the top prediction
            if idx == top1_id:
                st.write(f"🎯 **{c_name.capitalize()}**: {percentage:.2f}%")
            else:
                st.write(f"{c_name.capitalize()}: {percentage:.2f}%")

            st.progress(float(c_prob))

else:
    st.info("Please upload an image from the top-left area to test it.")
