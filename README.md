# Smart Store Product Detection System
### Retail Object Detection with YOLO, Streamlit, and OpenCV

<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Model-YOLO%20%7C%20Ultralytics-purple" alt="YOLO">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/Video%20Processing-OpenCV-green" alt="OpenCV">
</p>

## Overview

**Smart Store Product Detection System** is a computer vision application that detects retail products from images, uploaded videos, and a live webcam stream.

The application uses a custom-trained YOLO model (`best.pt`) to identify and localize products. It provides an interactive Streamlit interface for selecting an input source and viewing the detection results.

This project extends image-based object detection into video and webcam scenarios, demonstrating how a trained computer vision model can be integrated into an interactive application.

## Key Features

- **Image Detection:** Upload an image and detect retail products.
- **Video Detection:** Upload a video and run product detection on its frames.
- **Webcam Detection:** Use a browser webcam stream for live product detection.
- **Multi-Class Detection:** Identify product categories supported by the trained model.
- **Detection Visualization:** Display predicted labels and bounding boxes.
- **Interactive Interface:** Access detection modes through a Streamlit application.

## Supported Input Sources

| Input | Description |
|---|---|
| Image Upload | Detect products in an uploaded image |
| Video Upload | Detect products across frames of an uploaded video |
| Browser Webcam | Run detection on a live webcam stream using `streamlit-webrtc` |

## Model

The application uses a custom-trained YOLO object detection model saved as:

`best.pt`

The model detects products belonging to the classes used during training. For each detection, it predicts the product class and bounding-box location, along with a confidence score.

### Supported Product Classes

The model is trained to detect these retail product categories:

`cake`, `candy`, `cereal`, `chips`, `chocolate`, `coffee`, `fish`, `honey`, `jam`, `milk`, `oil`, `pasta`, `rice`, `soda`, `sugar`, `tea`, `vinegar`, `water`

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| YOLO / Ultralytics | Object detection and model inference |
| Streamlit | Interactive web interface |
| OpenCV | Image and video frame processing |
| streamlit-webrtc | Browser webcam streaming |

## Detection Workflow

```text
             Select Input
                 |
      +----------+----------+
      |          |          |
    Image      Video     Webcam
   Upload     Upload    (Browser)
      |          |          |
      +----------+----------+
                 |
                 v
       Image / Frame Input
                 |
                 v
        Custom YOLO Model
             best.pt
                 |
                 v
        Product Detection
                 |
                 v
   Class Labels + Bounding Boxes
                 |
                 v
       Results in Streamlit
```

## Getting Started

### Prerequisites

- Python installed
- Git
- A webcam for live webcam detection
- Browser permission to access the camera

### 1. Clone the Repository

```bash
git clone https://github.com/ashish-wandhare/object_detection_webcam.git
```

### 2. Navigate to the Project Directory

```bash
cd object_detection_webcam
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

The application entry point in this repository is `app2.py`.

```bash
streamlit run app2.py
```

Open the local URL displayed in the terminal to access the application.

## Using Webcam Detection

1. Open the application in your browser.
2. Select the webcam detection option.
3. Allow camera access when prompted by the browser.
4. Start the stream and view the detection results.

Browser webcam access may require HTTPS or localhost, depending on the browser and deployment environment. The user must grant camera permission.

## Repository Contents

| File | Description |
|---|---|
| `app2.py` | Streamlit application |
| `best.pt` | Custom-trained object detection model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Potential Applications

- Retail product identification
- Visual product inspection
- Webcam-based object detection demonstrations
- Video-based product detection
- Smart retail and shelf-monitoring prototypes

## Future Enhancements

- Product count summaries by detected class
- Adjustable confidence threshold
- Detection performance and processing metrics
- Improved real-time frame processing
- Inventory analytics and shelf-monitoring features
- Integration with a model-serving API

## Repository

[**object_detection_webcam**](https://github.com/ashish-wandhare/object_detection_webcam)

## Author

**Ashish Sudhakar Wandhare**

AI/ML Engineer | Python | Machine Learning | Computer Vision | Generative AI

[GitHub](https://github.com/ashish-wandhare)
