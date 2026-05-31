<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi" />
<img src="https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv" />
<img src="https://img.shields.io/badge/YOLOv8-ultralytics-FF6F00?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

# 🔵 BrailleVision

### Real-Time Braille Detection & Text-to-Speech Converter

**BrailleVision Hackathon 2026** | Built during: 31 May 4PM → 1 June 5PM IST

[**Live Demo**](#) · [**Report Bug**](#) · [**API Docs**](http://localhost:8000/docs)

</div>

---

## 📌 Problem Statement

Over **253 million people** worldwide live with visual impairment. Braille is the primary written language for blind individuals, yet there is no widely accessible, real-time tool to convert physical embossed Braille into digital text and speech.

**BrailleVision** solves this by combining computer vision (YOLOv8 + OpenCV) with text-to-speech to create a real-time, camera-based Braille reader — accessible to anyone with a smartphone or webcam.

---

## 🎯 Solution Overview

```
┌─────────────────────────────────────────────────────────┐
│                     BrailleVision                       │
│                                                         │
│  📷 Camera          🔬 Preprocessing      🤖 Detection │
│  ──────────  ──►   ────────────────  ──►  ──────────── │
│  Webcam /          OpenCV CLAHE           YOLOv8n or   │
│  Image Upload      Adaptive Threshold     Classical CV  │
│                    Noise Reduction                      │
│                                                         │
│  📝 Decode         🔊 TTS Output         🌐 Frontend   │
│  ──────────  ──►  ────────────────  ──►  ──────────── │
│  Dot Pattern       gTTS / pyttsx3        React-like    │
│  → Letter Map      MP3 Stream            Web UI        │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🎥 **Real-time webcam** Braille detection via WebSocket stream
- 📸 **Image upload** support (JPEG, PNG, WebP)
- 🤖 **YOLOv8** object detection (with classical CV fallback)
- 🔬 **OpenCV preprocessing** — CLAHE contrast enhancement, adaptive thresholding for embossed dots
- 🔊 **Text-to-Speech** output using gTTS
- 📖 **Interactive Braille reference chart** (A-Z with dot patterns)
- ⚡ **WebSocket live stream** endpoint
- 📊 **Confidence scoring** per detection
- 🌙 **Dark glassmorphism UI** — premium design
- 💾 **Export decoded text** as .txt file

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Detection | YOLOv8n (Ultralytics) | Braille cell detection |
| Preprocessing | OpenCV 4.9 | Image enhancement for embossed dots |
| Backend | FastAPI + Uvicorn | REST API + WebSocket server |
| Decoder | Custom Python (rule-based) | Braille dot pattern → letter |
| TTS | gTTS | Text-to-speech audio |
| Frontend | HTML/CSS/JavaScript | Premium real-time web UI |
| Dataset | Roboflow + Synthetic | Training data |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- Webcam (for live detection)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/braille-vision.git
cd braille-vision
```

### 2. Set up virtual environment
```bash
python -m venv venv

# Windows:
venv\bin\activate.bat

# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt

# Optional: Install YOLOv8
pip install ultralytics --trusted-host pypi.org
```

### 4. Generate sample dataset (optional)
```bash
cd backend
python generate_samples.py
```

### 5. Run the backend
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the frontend
Open your browser and navigate to:
```
http://localhost:8000/app/index.html
```

**OR** double-click `start.bat` (Windows) to launch everything at once.

---

## 🧪 Run Inference (CLI)

```bash
# Single image
python backend/inference_cli.py --image dataset/sample_inputs/hello_clean.jpg

# Save annotated output
python backend/inference_cli.py --image dataset/sample_inputs/hello_clean.jpg --output result.jpg

# JSON output (for scripting)
python backend/inference_cli.py --image dataset/sample_inputs/hello_clean.jpg --json

# Live webcam mode
python backend/inference_cli.py --webcam
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Backend status + model info |
| `/detect` | POST | Upload image → decoded text + annotated image |
| `/tts` | POST | Text → base64 MP3 audio |
| `/decode` | POST | Manual dot patterns → text |
| `/braille/reference` | GET | All A-Z Braille patterns |
| `/ws/live` | WebSocket | Real-time frame processing |
| `/docs` | GET | Swagger API documentation |

### Example: Detect Braille from Image
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@dataset/sample_inputs/hello_clean.jpg"
```

Response:
```json
{
  "success": true,
  "text": "hello",
  "confidence": 0.82,
  "method": "classical_cv",
  "cells_detected": 5,
  "dots_detected": 15,
  "annotated_image": "data:image/jpeg;base64,..."
}
```

### Example: Text-to-Speech
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "lang": "en"}'
```

---

## 📁 Project Structure

```
braille-vision/
├── backend/
│   ├── app.py                  # FastAPI main application
│   ├── braille_decoder.py      # Braille dot pattern → letter mapping
│   ├── inference.py            # YOLOv8 + OpenCV inference engine
│   ├── inference_cli.py        # Command-line inference tool
│   ├── generate_samples.py     # Synthetic dataset generator
│   ├── requirements.txt        # Python dependencies
│   └── models/
│       └── best.pt             # YOLOv8 trained weights (place here)
├── frontend/
│   └── index.html              # Premium single-page web application
├── training/
│   └── braille_train.ipynb     # Google Colab training notebook
├── dataset/
│   ├── data.yaml               # YOLO dataset config
│   ├── images/
│   │   ├── train/              # Training images
│   │   └── val/                # Validation images
│   ├── labels/
│   │   ├── train/              # YOLO format labels
│   │   └── val/
│   ├── sample_inputs/          # Sample Braille test images
│   └── sample_outputs/         # Annotated detection results
├── venv/                       # Python virtual environment
├── start.bat                   # Windows one-click launcher
└── README.md                   # This file
```

---

## 🤖 Model Details

### Architecture
- **Base Model**: YOLOv8n (nano) — fastest inference for real-time use
- **Input Size**: 640×640 pixels
- **Classes**: 26 (letters a-z)
- **Framework**: Ultralytics

### Training
- **Dataset**: Roboflow Braille detection dataset + synthetic augmented data
- **Epochs**: 100 with early stopping (patience=20)
- **Optimizer**: AdamW (lr=0.001)
- **Augmentation**: HSV shifts, rotation (±5°), mosaic, mixup, copy-paste

### Hyperparameters
```
epochs:       100
batch:        16
imgsz:        640
optimizer:    AdamW
lr0:          0.001
lrf:          0.01
momentum:     0.937
weight_decay: 0.0005
augment:      True
fliplr:       0.0  (Braille is direction-sensitive!)
```

### Fallback: Classical CV Pipeline
When `best.pt` is not present, the system uses:
1. **CLAHE** contrast enhancement
2. **Adaptive Gaussian threshold**
3. **SimpleBlobDetector** for dot finding
4. **Cell grid analysis** to map dots to Braille positions
5. **Rule-based decoder** to map patterns to letters

---

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Source | Roboflow Universe + Synthetic Generation |
| Format | YOLOv8 (YOLO format labels) |
| Classes | 26 (a-z Braille letters) |
| Train/Val/Test split | 80% / 10% / 10% |
| Annotation Format | YOLO (cx cy w h normalized) |
| Preprocessing | CLAHE, adaptive threshold, augmentation |

### Class Names
```
['a','b','c','d','e','f','g','h','i','j','k','l','m',
 'n','o','p','q','r','s','t','u','v','w','x','y','z']
```

### Dataset Download
📦 Dataset available at: [Roboflow Link] or `python backend/generate_samples.py` for synthetic samples.

---

## 🎥 Demo

> Demo video shows real physical Braille paper being detected by webcam in real-time.

**Steps shown in demo:**
1. Physical embossed Braille paper placed in front of camera
2. BrailleVision web UI opens
3. "Capture & Detect" clicked — detection boxes appear on cells
4. Decoded English text appears with typewriter animation
5. "Speak" button pressed — text read aloud
6. Export to text file demonstrated

---

## 🔬 How It Works

### Preprocessing Pipeline (OpenCV)
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)
binary = cv2.adaptiveThreshold(enhanced, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
```

### Braille Dot Numbering
```
Left col  Right col
  1   4
  2   5
  3   6
```

### Cell Decoding Example
Dots {1, 2, 5} → 'h' (positions 1, 2, 5 are raised)

---

## 🏆 Built At

**BrailleVision Hackathon 2026**
- Build Timeline: 31 May 4:00 PM IST → 1 June 5:00 PM IST
- All commits verifiable within this window

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for accessibility | BrailleVision Hackathon 2026

</div>
