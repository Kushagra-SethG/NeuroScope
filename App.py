import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from datetime import datetime
import torch

# Fix for PyTorch 2.6 weights_only issue
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.tasks.SegmentationModel', 
    'ultralytics.nn.tasks.ClassificationModel',
    'ultralytics.nn.tasks.PoseModel',
    'ultralytics.nn.modules.conv.Conv',
    'ultralytics.nn.modules.block.C2f',
    'ultralytics.nn.modules.head.Detect'
])

st.set_page_config(
    page_title="NeuroScope: AI Brain Tumor Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================
# Sidebar
# ============================  
with st.sidebar:
    st.title("🧠 NeuroScope")
    st.markdown("""
    **Instructions:**
    - Upload an MRI/CT scan image (JPG, JPEG, PNG).
    - Wait for the model to analyze the image.
    - View the detection result and download the annotated image.
    """)
    st.markdown("---")
    st.markdown("**Model:** YOLOv8n (Ultralytics)")
    st.markdown("**App by:** Kushagra Agrawal")
    st.markdown("**Date:** {}".format(datetime.now().strftime("%B %d, %Y")))
    st.markdown("---")
    st.info("This app is for educational/demo purposes only. Not for clinical use.")

# ============================
# Load YOLOv8 Model
# ============================
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")   # Change path if needed
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("Failed to load YOLO model. Please check your model file.")
    st.stop()

# ============================
# Streamlit UI
# ============================

st.markdown("""
# 🧠 NeuroScope: AI Brain Tumor Detector
Detect brain tumors in MRI/CT images using state-of-the-art YOLOv8 deep learning.
""")

st.markdown(
    "<span style='color:gray'>Upload an MRI/CT scan image to detect brain tumors. The model will tell any detected tumors in the image.</span>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("**Step 1:** Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Step 2: Uploaded Image Preview")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save to temporary file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name

    st.markdown("---")
    st.subheader("Step 3: Detection Progress")
    progress = st.progress(0, text="Initializing model...")
    progress.progress(20, text="Running inference...")
    results = model.predict(source=tmp_path, conf=0.25)
    progress.progress(80, text="Processing results...")

    # Check if any tumors detected
    boxes = results[0].boxes
    progress.progress(100, text="Detection complete!")
    st.markdown("---")
    st.subheader("Step 4: Detection Result")
    
    if boxes is not None and len(boxes) > 0:
        st.error("⚠️ TUMOR DETECTED!")
        st.markdown("""
        ### Important Notice:
        - This is a preliminary screening result
        - Please consult a medical professional immediately
        - This tool is for educational purposes only
        """)
    else:
        st.success("✅ No tumor detected in the image")
        st.info("Note: Always consult healthcare professionals for medical advice.")

    # Display the original image
    st.image(image, caption="Analyzed Image", use_container_width=True)

    st.markdown("---")
    st.warning("Remember: This is an educational tool and should not be used for clinical diagnosis.")
