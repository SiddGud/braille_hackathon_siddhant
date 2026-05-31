# BrailleVision
Real-Time Optical Braille Recognition and Translation System

## 1. Problem Statement
According to the World Health Organization, at least 2.2 billion people globally have a near or distance vision impairment. While Braille remains the fundamental tool for literacy, education, and employment for the visually impaired, Braille literacy rates have been declining. A major contributing factor is the lack of accessible translation tools. Sighted teachers, parents, and peers often cannot read Braille, creating a significant communication barrier. Traditional Braille translation requires expensive, specialized hardware or manual transcription, which is slow and inaccessible to the general public.

## 2. Solution Overview
BrailleVision bridges this communication gap by democratizing Braille translation. It is an end-to-end, real-time computer vision system that translates physical, embossed Braille into digital text and audible speech using any standard web camera or smartphone camera. By combining state-of-the-art deep learning object detection with classical computer vision techniques, BrailleVision achieves high accuracy across various lighting conditions, paper types, and camera angles, making Braille instantly readable for anyone.

## 3. System Architecture

The system operates on a highly optimized, low-latency client-server architecture designed for real-time video stream processing.

```text
[ Client Application ]                      [ Backend API (FastAPI) ]
       |                                              |
       |--- 1. Capture Video Frame (Base64) --------->|
       |                                              |--- 2. Preprocessing Pipeline
       |                                              |      (CLAHE, Grayscale, Blur)
       |                                              |
       |                                              |--- 3. YOLOv8 Detection Engine
       |                                              |      (Bounding Box Localization)
       |                                              |
       |                                              |--- 4. Hybrid Detection Engine
       |                                              |      (Classical CV Fallback)
       |                                              |
       |                                              |--- 5. Spatial Sorting & Decoding
       |                                              |      (XY clustering, Binary to Text)
       |                                              |
       |<-- 6. Return JSON (Text + Bounding Boxes) ---|
       |                                              |
[ Render UI & Trigger Text-to-Speech ]
```

## 4. Hybrid Detection Engine
The key innovation of BrailleVision is its Hybrid Detection Engine. While modern deep learning (YOLO) is exceptional at generalized object detection, Braille characters are extremely uniform (2x3 dot grids) and rely entirely on minute spatial differences. A standard CNN spatial pooling layer often confuses mirrored characters (e.g., 'w' and 'r') when lighting directions change. 

To solve this, our Hybrid Engine utilizes a two-stage approach:
1. **Macro-Localization:** The YOLOv8 neural network is tasked solely with finding the bounding boxes of individual Braille cells, ignoring the specific character classification.
2. **Micro-Extraction:** Once the bounding box is isolated, the engine falls back to deterministic, classical computer vision algorithms to mathematically calculate the presence of dots at the 6 specific grid coordinates within that isolated cell. 

This hybrid approach ensures that the system benefits from the robustness of deep learning for localization, while maintaining the mathematical precision of classical algorithms for actual dot extraction.

## 5. Model Details
- **Base Architecture:** YOLOv8n (Nano) 
- **Fine-Tuning:** The model was fine-tuned for 50 epochs using a highly constrained augmentation strategy. 
- **Crucial Hyperparameters:** `fliplr=0.0` and `flipud=0.0`. Horizontal and vertical flipping augmentations were explicitly disabled during training. Braille is chiral; flipping a character horizontally changes its fundamental meaning. Disabling these augmentations prevented catastrophic class confusion during training.
- **Batch Size:** 8 (optimized for lower-tier hardware constraints).
- **Final Metrics:** Achieved an mAP50 of 0.989 on the validation set.
- **Training Hardware:** NVIDIA T4 Tensor Core GPU.

## 6. Preprocessing Pipeline
Embossed Braille relies entirely on shadows to be visible. Different lighting angles drastically change the visual representation of the dots. The preprocessing pipeline ensures normalization before tensor conversion:
1. **Grayscale Conversion:** Removes chromatic noise, forcing the model to rely strictly on luminance (shadows and highlights).
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Unlike global histogram equalization which amplifies noise, CLAHE divides the image into tiles and enhances contrast locally. This is vital for paper that is slightly bent or unevenly lit.
3. **Gaussian Blur:** A slight 3x3 kernel blur is applied to reduce high-frequency paper grain noise without destroying the crisp edges of the embossed dots.

## 7. Classical CV Fallback (Mathematical Implementation)
When the neural network confidence falls below a threshold, the system triggers the classical fallback mechanism on the cropped cell.
1. **Adaptive Gaussian Thresholding:** Instead of a global threshold (which fails on gradient lighting), the threshold value $T(x,y)$ is calculated as the weighted sum (cross-correlation with a Gaussian window) of the $11 \times 11$ neighborhood around $(x,y)$, minus a constant $C$.
2. **Contour Filtering:** Topological structural analysis algorithms find contours in the binary image. Contours are filtered by area $A$ where $A_{min} < A < A_{max}$ to eliminate speckle noise and retain only legitimate Braille dots.
3. **Grid Mapping:** The remaining centroids are mapped to a 2x3 matrix, converted to a 6-bit binary string, and matched against the standard Braille unicode dictionary.

## 8. Dataset Details

| Dataset Name | Source | Purpose | Volume |
|---|---|---|---|
| Angelina Braille Dataset | Roboflow / Open Source | Primary training data. Contains complex real-world shadows and warped paper. | ~1,500 images |
| DSBI (Double-Sided) | Academic Repository | Edge-case training for double-sided punched paper interference. | ~300 images |
| Synthetic Generator | Custom Python Script | Baseline bootstrapping and perfect-condition validation matrices. | Dynamically Generated |

## 9. Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (Vanilla), JavaScript, Canvas API |
| **Backend API** | Python 3.11, FastAPI, Uvicorn |
| **Machine Learning** | Ultralytics (YOLOv8), PyTorch |
| **Computer Vision** | OpenCV (cv2), NumPy |
| **Audio Processing** | gTTS (Google Text-to-Speech), Pygame |

## 10. API Reference

### `GET /health`
Verifies backend status.
- **Response:** `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### `POST /detect`
Processes a base64 encoded image frame and returns detected characters and bounding boxes.
- **Body:**
```json
{
  "image": "base64_encoded_string_here..."
}
```
- **Response:** `200 OK`
```json
{
  "text": "hello world",
  "confidence": 0.94,
  "method": "yolo",
  "cells": [
    {
      "letter": "h",
      "x": 120,
      "y": 45,
      "w": 30,
      "h": 45,
      "confidence": 0.98
    }
  ]
}
```

### `POST /speak`
Generates and streams audio for the provided text.
- **Body:**
```json
{
  "text": "hello world"
}
```
- **Response:** `200 OK` (Audio/MPEG Stream)

## 11. Project Structure

```text
BrailleVision/
├── backend/
│   ├── app.py                 # FastAPI server and endpoint routing
│   ├── inference.py           # YOLO inference and Classical CV fallback logic
│   ├── generate_samples.py    # Synthetic dataset generator script
│   ├── requirements.txt       # Python dependencies
│   └── models/
│       └── best.pt            # Fine-tuned YOLOv8 neural network weights
├── frontend/
│   └── index.html             # Client UI, webcam capture, and canvas rendering
├── dataset/
│   ├── data.yaml              # YOLO dataset configuration map
│   └── sample_inputs/         # Validation images for testing
├── train.py                   # Automated Colab/Local training script
├── start.bat                  # Windows automated deployment script
└── README.md                  # Project documentation
```

## 12. Setup Instructions

### Windows (Automated)
1. Ensure Python 3.11+ is installed and added to your system PATH.
2. Clone this repository.
3. Double click `start.bat`. This script will automatically create a virtual environment, install all dependencies, start the backend server, and open the frontend in your default browser.

### Mac / Linux (Manual)
1. Ensure Python 3.11+ is installed.
2. Open a terminal and clone the repository:
   ```bash
   git clone https://github.com/YourUsername/braille_hackathon.git
   cd braille_hackathon
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

## 13. How to Run

If you did not use the automated `start.bat` script, you can run the system manually:

1. **Start the Backend:**
   ```bash
   cd backend
   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Open the Frontend:**
   Simply double-click `frontend/index.html` in your file explorer to open it in any modern web browser.

## 14. Testing for Judges

To verify the system's capabilities without a webcam or physical Braille paper:
1. Open the frontend application in your browser.
2. Under the camera feed, select the **Upload Image** tab.
3. Navigate to the `dataset/sample_inputs/` folder in the project directory.
4. Upload `hello_clean.jpg` or `braille_embossed.jpg`.
5. The system will process the image, draw bounding boxes around the identified cells, and display the translated text in the results panel.
6. Click the speaker icon to test the Text-to-Speech integration.

## 15. Training Reproduction Steps

To verify or improve the model accuracy, you can reproduce our training pipeline using Google Colab:
1. Create a new Google Colab notebook and select a T4 GPU runtime.
2. Install dependencies: `!pip install ultralytics roboflow`
3. Download the dataset via the Roboflow API (requires a free API key):
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_KEY")
   project = rf.workspace().project("braille-character-detection")
   dataset = project.version(1).download("yolov8")
   ```
4. Copy the contents of `train.py` into a new Colab cell. Ensure `batch=8` and `cache=False` remain set to prevent RAM exhaustion.
5. Execute the cell. Training will complete in approximately 15-20 minutes. The output weights can be found in `runs/detect/train/weights/best.pt`.

## 16. Known Limitations and Future Improvements

- **Shadow Directionality Bias:** While mitigated by the hybrid engine and dataset augmentation, extreme variations in lighting direction (e.g., lighting from the bottom edge instead of the top) can still invert dot perception. Future iterations will implement automated image rotation based on localized shadow gradient mapping prior to inference.
- **Double-Sided Interference:** Standard embossed Braille paper often has dots punched on both sides. The reverse side creates "concave" dots that appear darker. The current model occasionally attempts to parse these concave dots as valid cells.
- **Mobile Native Application:** The current implementation relies on a web browser. Migrating the model to ONNX or CoreML for on-device inference via a native iOS/Android application would significantly reduce latency and eliminate the need for internet connectivity.

## 17. Disclosure

- **Datasets:** Models were trained utilizing open-source community data, specifically leveraging variations of the Angelina Braille Dataset hosted on Roboflow Universe.
- **Architecture:** The core object detection utilizes the Ultralytics YOLOv8 architecture. 
- **Generative AI:** Generative language models were utilized to assist in writing boilerplate CSS, scaffolding FastAPI endpoints, and drafting sections of this documentation. All core algorithmic logic, hybrid engine design, and model tuning were conceptualized and implemented manually.

## 18. License

This project is licensed under the MIT License - see the LICENSE file for details.
