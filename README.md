# 🧠 NeuroScope: AI Brain Tumor Detector

A state-of-the-art web application that uses YOLOv8 deep learning to detect brain tumors in MRI/CT scan images. Built with Streamlit for an intuitive user interface.

## 🚀 Features

- **AI-Powered Detection**: Uses YOLOv8n model for accurate brain tumor detection
- **Simple Interface**: Clean, user-friendly Streamlit web interface
- **Real-time Analysis**: Upload and analyze medical images instantly
- **Educational Tool**: Designed for learning and demonstration purposes
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technologies Used

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Frontend**: Streamlit
- **Image Processing**: PIL (Python Imaging Library)
- **Backend**: Python
- **Version Control**: Git with LFS for large model files

## 📋 Requirements

```
torch>=2.0.0,<2.6.0
torchvision>=0.15.0
ultralytics
streamlit
Pillow
pandas
opencv-python-headless
```

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kushagra-SethG/NeuroScope.git
   cd NeuroScope
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run App.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### 🌐 Live Demo

Access the live application: [NeuroScope on Streamlit Cloud](https://your-app-url.streamlit.app)

## 📱 How to Use

1. **Upload Image**: Choose an MRI or CT scan image (JPG, JPEG, PNG)
2. **Wait for Analysis**: The AI model will process your image
3. **View Results**: See if a tumor is detected or not
4. **Medical Consultation**: Always consult healthcare professionals for medical advice

## 📁 Project Structure

```
NeuroScope/
├── App.py                 # Main Streamlit application
├── yolov8n.pt            # YOLOv8 model weights (via Git LFS)
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python runtime version
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .gitattributes        # Git LFS configuration
└── README.md             # Project documentation
```

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This application is designed for educational and demonstration purposes
- **Not for Clinical Use**: Do not use for actual medical diagnosis
- **Consult Professionals**: Always seek advice from qualified healthcare professionals
- **No Medical Advice**: This tool does not provide medical advice or replace professional consultation

## 🔧 Configuration

The application uses YOLOv8n model with the following settings:
- **Confidence Threshold**: 0.25
- **Image Formats**: JPG, JPEG, PNG
- **Model**: YOLOv8n (optimized for speed and efficiency)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Acknowledgments

- **Ultralytics** for the YOLOv8 model
- **Streamlit** for the web framework
- **PyTorch** for deep learning capabilities
- Medical imaging community for research and datasets

## 📞 Contact

**Developer**: Kushagra Agrawal  
**GitHub**: [@Kushagra-SethG](https://github.com/Kushagra-SethG)  
**Project Link**: [https://github.com/Kushagra-SethG/NeuroScope](https://github.com/Kushagra-SethG/NeuroScope)

---

### 🔬 Technical Details

**Model Information**:
- Architecture: YOLOv8n (nano version for efficiency)
- Framework: PyTorch
- Input: RGB images (any resolution, auto-resized)
- Output: Bounding boxes for detected tumors

**Performance**:
- Inference Time: < 2 seconds on CPU
- Model Size: ~6.5 MB
- Supported Formats: JPG, JPEG, PNG

---

*Made with ❤️ for advancing AI in healthcare education*
