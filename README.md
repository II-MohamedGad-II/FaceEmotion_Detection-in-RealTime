# 🎭 Real-time Face Emotion Detection

A real-time emotion detection application built with **Streamlit** and **OpenCV** that analyzes facial expressions through your webcam and classifies emotions using a trained deep learning model.

## 🌟 Features

- **📹 Real-time Processing**: Live webcam emotion detection
- **🎯 High Accuracy**: Uses a trained CNN model for emotion classification  
- **👤 Face Detection**: OpenCV Haar Cascade for reliable face detection
- **🎨 Interactive UI**: Clean and responsive Streamlit web interface
- **📊 Confidence Scores**: Shows prediction confidence for each emotion
- **🚀 Easy Setup**: Simple installation and usage

## 🎬 Demo

The application provides real-time emotion detection with:
- Live camera feed display
- Face bounding boxes around detected faces
- Emotion labels with confidence scores
- Start/Stop camera controls
- Real-time status updates

## 🛠️ Technology Stack

- **Python 3.10+**: Core programming language
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning model inference
- **OpenCV**: Computer vision and face detection
- **NumPy**: Numerical computations

## 📋 Supported Emotions

The model can detect 4 different emotions:

| Emotion | Description | Example Use Case |
|---------|-------------|------------------|
| 😊 **Happy** | Joy, satisfaction, positive mood | Customer satisfaction analysis |
| 😐 **Neutral** | Calm, expressionless, relaxed | Baseline emotional state |
| 😢 **Sad** | Sadness, disappointment, sorrow | Mental health monitoring |
| 😮 **Surprise** | Shock, amazement, unexpected | Reaction analysis |

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/II-MohamedGad-II/FaceEmotion_Detection-in-RealTime.git
cd FaceEmotion_Detection-in-RealTime
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv .venv


# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run EmoDetection.py
```

### 5. Open Browser
Navigate to `http://localhost:8501` and click **"Start Camera"**

## 📁 Project Structure

```
FaceEmotion_Detection-in-RealTime/
├── EmoDetection.py          # Main Streamlit application
├── model.h5                 # Trained emotion classification model
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## 🔧 How It Works

1. **Camera Access**: Application requests webcam permission
2. **Frame Capture**: Continuous video frame capture from camera
3. **Face Detection**: OpenCV Haar Cascade identifies faces in frames
4. **Preprocessing**: Detected faces are:
   - Converted to grayscale
   - Resized to 48x48 pixels
   - Normalized (pixel values 0-1)
   - Reshaped for model input (1, 48, 48, 1)
5. **Emotion Prediction**: CNN model predicts emotion probabilities
6. **Visualization**: Results displayed with bounding boxes and labels

## ⚙️ Technical Details

### Model Configuration
The application uses a pre-trained Keras model (`model.h5`) that expects:
- **Input Shape**: (48, 48, 1) - Grayscale images
- **Output Classes**: 4 emotions (happy, neutral, sad, surprise)
- **Architecture**: Convolutional Neural Network
- **File Size**: ~2-5 MB

### Face Detection Settings
```python
# Haar Cascade parameters (adjustable in EmoDetection.py)
scaleFactor=1.1      # Image pyramid scaling factor
minNeighbors=5       # Minimum neighbor rectangles for detection
minSize=(30, 30)     # Minimum face size in pixels
```

### Performance Optimization
- Processes frames in real-time (~15-30 FPS)
- Efficient face detection with OpenCV
- Lightweight model for fast inference
- Minimal memory footprint

## 🛠️ Customization

### Adding New Emotions
To expand emotion detection:
1. Retrain the model with additional emotion classes
2. Update the `labels` list in `EmoDetection.py`:
   ```python
   labels = ['happy', 'neutral', 'sad', 'surprise', 'angry', 'fearful']
   ```
3. Replace `model.h5` with your new trained model

### Adjusting Detection Sensitivity
Modify parameters in `EmoDetection.py`:
```python
faces = face_cascade.detectMultiScale(
    gray_frame,
    scaleFactor=1.1,    # Lower = more sensitive (1.05-1.3)
    minNeighbors=5,     # Lower = more detections (3-8)
    minSize=(30, 30)    # Smaller = detect smaller faces
)
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**🎥 Camera Not Working**
- Ensure camera permissions are granted in browser/OS
- Close other applications using the camera
- Try different camera index: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**📁 Model Loading Error**
```
❌ Error loading emotion model: [Errno 2] No such file or directory: 'model.h5'
```
- Verify `model.h5` exists in the project root directory
- Check file permissions
- Ensure the model file isn't corrupted

**📦 Import/Dependency Errors**
```python
AttributeError: _ARRAY_API not found
```
- Activate virtual environment: `.venv\Scripts\activate`
- Install compatible NumPy: `pip install "numpy<2.0"`
- Reinstall OpenCV: `pip install opencv-python --force-reinstall`

**⚡ Performance Issues**
- Reduce camera resolution in code
- Process every 2nd or 3rd frame instead of all frames
- Close unnecessary applications
- Ensure adequate system resources

## 📊 Performance Metrics

- **Frame Rate**: 15-30 FPS (depending on hardware)
- **Model Inference**: <50ms per face
- **Memory Usage**: 200-500 MB during operation
- **Model Size**: ~2-5 MB
- **Supported Resolutions**: 480p to 1080p

## 🔒 Privacy & Security

- **🏠 Local Processing**: All processing happens on your local machine
- **🚫 No Data Collection**: No images or personal data are stored
- **📷 Camera Access**: Only active during detection sessions
- **🌐 Offline Capable**: Works without internet connection
- **🔐 Secure**: No external data transmission

## 🎯 Use Cases & Applications

- **🎓 Educational**: Learn computer vision and deep learning
- **🔬 Research**: Emotion analysis and behavioral studies  
- **💼 Portfolio**: Showcase AI/ML development skills
- **🧪 Prototyping**: Foundation for emotion-aware applications
- **🎮 Interactive**: Gaming and entertainment applications
- **📊 Analytics**: User experience and engagement analysis

## 🚀 Future Enhancements

- [ ] Support for multiple simultaneous faces
- [ ] Emotion history and analytics dashboard
- [ ] Export detection results to CSV/JSON
- [ ] Additional emotion classes (anger, fear, disgust)
- [ ] Mobile-responsive interface
- [ ] Real-time emotion trend visualization
- [ ] Integration with external APIs
- [ ] Custom model training interface

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? Submit a detailed bug report
- 💡 **Feature Requests**: Have an idea? Suggest new functionality
- 🔧 **Code Contributions**: Submit pull requests with improvements
- 📚 **Documentation**: Help improve guides and examples
- 🧪 **Testing**: Test on different platforms and configurations

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper comments
4. Test thoroughly on your local machine
5. Update documentation if needed
6. Submit a pull request with clear description

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohamed Gad**
- GitHub: [@II-MohamedGad-II](https://github.com/II-MohamedGad-II)
- LinkedIn: https://www.linkedin.com/in/mohamed-gad-970a74280/
- Portfolio: loading..

## 🙏 Acknowledgments

- **OpenCV Community** for robust computer vision tools
- **TensorFlow Team** for the powerful deep learning framework
- **Streamlit Creators** for the amazing web app framework
- **Open Source Community** for continuous inspiration and support

---

**⭐ Star this repository if you found it helpful!**  
**🍴 Fork it to create your own emotion detection app!**

