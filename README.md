# BrailleVision 👁️
**AI-powered real-time Braille recognition for an inclusive world.**

BrailleVision is a full-stack computer vision web application that detects physical Braille text via a live webcam stream or image upload, repairs OCR anomalies using a fuzzy-matching heuristic pipeline, and converts the decoded text to spoken audio using on-device TTS.

> **Submission for: BrailleVision Hackathon 2026**
> Built with ❤️ — Empowering Independence

---

## 📁 Repository Structure

```
braille_hackathon_siddhant/
├── README.md
├── requirements.txt
├── setup_instructions.md
├── ai_tools_disclosure.md
│
├── frontend/
│   ├── index.html          # Main UI (single-page app)
│   ├── temp.js             # All JS logic (WebRTC, WebSocket, TTS)
│   └── logo.png            # App logo
│
├── backend/
│   ├── app.py              # FastAPI server + WebSocket handler
│   └── inference.py        # YOLO inference + heuristic text repair
│
├── model/
│   ├── best.pt             # Trained YOLO weights (52MB)
│   └── model_info.md       # Architecture and load instructions
│
├── training/
│   ├── yolo_train_command.txt    # Exact training command used
│   ├── training_logs/            # Training logs (add your screenshots here)
│   └── results/                  # Confusion matrix, mAP curves (add here)
│
├── dataset/
│   ├── data.yaml                 # YOLO dataset configuration
│   ├── dataset_info.md           # Full dataset details and links
│   ├── sample_images/            # Place sample dataset images here
│   └── sample_annotations/       # Place sample label .txt files here
│
├── sample_inputs/                # Test Braille images for judges to run inference
├── sample_outputs/               # Expected outputs for the sample inputs
│
└── demo/
    ├── demo_video_link.txt       # Link to the demo video
    └── screenshots/              # Screenshots of the running app
```

---

## ⚡ Quick Start (For Judges)

### 1. Clone & Install
```bash
git clone https://github.com/SiddGud/braille_hackathon_siddhant.git
cd braille_hackathon_siddhant
pip install -r requirements.txt
```

### 2. Run the Backend
```bash
python backend/app.py
# Backend starts on http://localhost:8000
```

### 3. Open the Frontend
Open `frontend/index.html` in your browser (or serve with `python -m http.server 3000` from the `frontend/` folder).

### 4. Run Inference on a Sample Image
```bash
python inference/inference.py --source sample_inputs/test_braille.jpg --weights model/best.pt
```

---

## ⚠️ CRITICAL: Image Input Requirements

BrailleVision uses shadow-based computer vision. The quality of the input photo directly determines accuracy.

### Rule 1: Use the BACK of the paper
The model was trained on **indented craters** (the back side of punched Braille paper).
- ✅ **DO:** Flip the paper so dots appear as small indented holes.
- ❌ **DON'T:** Use the front side with raised bumps.

### Rule 2: Directional Side Lighting
- ✅ **DO:** Shine a phone flashlight at a **low angle from the bottom-left corner** of the paper.
- ❌ **DON'T:** Use overhead room lighting or a ceiling lamp — it destroys all shadows.

---

## 🧠 How BrailleVision Works

| Step | Component | Description |
|------|-----------|-------------|
| 1 | WebRTC Capture | 1280×720 live video streamed as base64 over WebSocket at 2 FPS |
| 2 | Preprocessing | OpenCV CLAHE contrast enhancement |
| 3 | AI Detection | YOLOv8-nano classifies 26 Braille character classes |
| 4 | Text Repair | Fuzzy-match heuristic engine corrects catastrophic YOLO misclassifications |
| 5 | TTS Output | Decoded words spoken aloud via Web Speech API |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | YOLOv8 Nano (Ultralytics) |
| CV Pipeline | OpenCV, CLAHE |
| Backend | Python, FastAPI, WebSockets |
| Frontend | HTML5, Vanilla JS, WebRTC |
| Audio | Web Speech API (on-device TTS) |

---

## 📦 Model Access

The trained weights are included directly in this repository at `model/best.pt` (52MB).

For additional verification, see `model/model_info.md`.

---

## 📊 Dataset

Full dataset details are in `dataset/dataset_info.md`.

---

## 🤖 AI Tools Disclosure

See `ai_tools_disclosure.md` for a full list of AI tools used during development.

---

*© 2025 BrailleVision · Empowering Independence*
