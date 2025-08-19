import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from datetime import datetime

st.set_page_config(
    page_title="NeuroScope: AI Brain Tumor Detector",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================
# Sidebar
# ============================  
with st.sidebar:
    st.title("� NeuroScope")
    st.markdown("""
    **Instructions:**
    - Upload an MRI/CT scan image (JPG, JPEG, PNG).
    - Wait for the model to analyze the image.
    - View the detection result and download the annotated image.
    """)
    st.markdown("---")
    st.markdown("**Model:** YOLOv8n (Ultralytics)")
    st.markdown("**App by:** Your Name")
    st.markdown("**Date:** {}".format(datetime.now().strftime("%B %d, %Y")))
    st.markdown("---")
    st.info("This app is for educational/demo purposes only. Not for clinical use.")

# ============================
# Load YOLOv8 Model
# ============================
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # Change path if needed
    return model

model = load_model()

# ============================
# Streamlit UI
# ============================


st.markdown("""
#  NeuroScope: AI Brain Tumor Detector
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

    # Check if any tumors detected (YOLO returns boxes in results[0].boxes)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.error("⚠️ Tumor detected! Please consult a medical professional.")
        st.markdown(f"**Number of detected regions:** {len(boxes)}")
    else:
        st.success("✅ No tumor detected.")

    # Save and display result
    result_img_path = os.path.join("result.jpg")
    results[0].save(filename=result_img_path)

    progress.progress(100, text="Detection complete!")
    st.markdown("---")
    st.subheader("Step 4: Detection Result")
    st.image(result_img_path, caption="Detection Result", use_container_width=True)

    with open(result_img_path, "rb") as f:
        st.download_button(
            label="Download Annotated Image",
            data=f,
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )

    st.info("Detection complete! For best results, use high-quality MRI/CT images.")
