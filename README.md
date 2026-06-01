# BrailleVision
Real-Time Optical Braille Recognition and Translation System

1. Problem Statement
According to the World Health Organization, at least 2.2 billion people globally have a near or distance vision impairment. While Braille remains the fundamental tool for literacy, education, and employment for the visually impaired, Braille literacy rates have been declining. A major contributing factor is the lack of accessible translation tools. Sighted teachers, parents, and peers often cannot read Braille, creating a significant communication barrier. Traditional Braille translation requires expensive, specialized hardware or manual transcription, which is slow and inaccessible to the general public.

2. Solution Overview
BrailleVision bridges this communication gap by democratizing Braille translation. It is an end‑to‑end, real‑time computer vision system that translates physical, embossed Braille into digital text and audible speech using any standard web camera or smartphone camera. By leveraging state‑of‑the‑art deep learning object detection, BrailleVision achieves high accuracy across various lighting conditions, paper types, and camera angles, making Braille instantly readable for anyone.

3. System Architecture
The system operates on a highly optimized, low‑latency client‑server architecture designed for real‑time video stream processing.

[ Client Application ]                      [ Backend API (FastAPI) ]
       |                                              |
       |--- 1. Capture Video Frame (Base64) --------->|
       |                                              |--- 2. YOLOv8 Detection Engine
       |                                              |      (Single-stage localization
       |                                              |       and classification)
       |                                              |
       |                                              |--- 3. Spatial Sorting & Decoding
       |                                              |      (XY clustering, Binary to Text)
       |                                              |
       |                                              |--- 4. Confidence Filtering
       |                                              |      (Thresholding weak detections)
       |                                              |
       |<-- 5. Return JSON ( Text + Bounding Boxes ) ---|
       |                                              |
[ Render UI & Trigger Text‑to‑Speech ]

4. The Engineering Journey & Pipeline
During development, we initially experimented with taking heavier, pre-existing models (including 11M+ parameter YOLO models and the open-source DotNeuralNet weights) and fine-tuning them across multiple GPU platforms (Local NVIDIA T4s and Google Colab). While these heavy models reached impressive validation metrics (up to ~93% mAP), we found they were incredibly brittle in the real world—hallucinating wildly on our specific physical test images due to slight lighting variations.

We realized that raw model size was not the answer. Instead, we shifted our architecture to use a balanced foundation model combined with a heavily optimized, multi-stage pipeline:

* **Stage 1 (OpenCV Preprocessing):** Before the image even touches the neural network, we apply a mathematical **Bilateral Filter** (`cv2.bilateralFilter`). This actively smooths out paper grain, shadows, and noise, while preserving the sharp, high-frequency edges of the physical braille dots.
* **Stage 2 (Test-Time Augmentation):** We run the YOLO inference with **TTA (Test-Time Augmentation)** enabled, allowing the model to internally scale and evaluate the image at multiple resolutions.
* **Stage 3 (Foundation Model):** We utilized the open-source DotNeuralNet YOLOv8-m weights (pretrained on the Angelina dataset) as our underlying baseline brain.
* **Stage 4 (Post-Processing Heuristics):** Since YOLO models will inevitably make character-level errors on blurry images, we implemented a custom Python `difflib` autocorrection layer that acts as a spellchecker, evaluating the spatial output and fuzzy-matching it to physical context.

5. Model Details
**Base Architecture:** YOLOv8-m (Foundation Model via DotNeuralNet)  
**Preprocessing Layer:** OpenCV Bilateral Filtering (d=9, sigmaColor=75, sigmaSpace=75) + Test-Time Augmentation (TTA)  
**Post-Processing:** Custom Spatial Clustering + `difflib` Fuzzy Context Matching  
**Final Metrics:** Achieved high accuracy on both generalized textbook scans and our physical hackathon test images.  

6. Dataset Details
| Dataset Name | Source | Purpose |
| --- | --- | --- |
| Angelina Braille Dataset (via Foundation Weights) | Open Source | Provided the generalized baseline for textbook and document scanning. |
| Kaggle Braille Character Dataset | Kaggle | Initial baseline bootstrapping and experimentation. |
| Custom Physical Images | Local | Used to tune our mathematical preprocessing and autocorrect layers. |

8. Technology Stack
Layer	Technology  
Frontend	HTML5, CSS3 (Vanilla), JavaScript, Canvas API  
Backend API	Python 3.11, FastAPI, Uvicorn  
Machine Learning	Ultralytics (YOLOv8), PyTorch  
Audio Processing	gTTS (Google Text‑to‑Speech), Pygame

9. API Reference
GET /health  
Verifies backend status.

Response: 200 OK  
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}

POST /detect  
Processes a base64 encoded image frame and returns detected characters and bounding boxes.

Body:
{
  "image": "base64_encoded_string_here..."
}
Response: 200 OK  
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

POST /speak  
Generates and streams audio for the provided text.

Body:
{
  "text": "hello world"
}
Response: 200 OK (Audio/MPEG Stream)

10. Project Structure
BrailleVision/
├── backend/
│   ├── app.py                 # FastAPI server and endpoint routing
│   ├── inference.py           # YOLO inference and decoding logic
│   ├── generate_samples.py    # Synthetic dataset generator script
│   ├── requirements.txt       # Python dependencies
│   └── models/
│       └── best.pt            # Fine‑tuned YOLOv8 neural network weights
├── frontend/
│   └── index.html             # Client UI, webcam capture, and canvas rendering
├── dataset/
│   ├── data.yaml              # YOLO dataset configuration map
│   └── sample_inputs/         # Validation images for testing
├── train.py                   # Automated Colab/Local training script
├── start.bat                  # Windows automated deployment script
└── README.md                  # Project documentation

11. Setup Instructions
Windows (Automated)  
Ensure Python 3.11+ is installed and added to your system PATH.  
Clone this repository.  
Double‑click `start.bat`. This script will automatically create a virtual environment, install all dependencies, start the backend server, and open the frontend in your default browser.

Mac / Linux (Manual)  
Ensure Python 3.11+ is installed.  
Open a terminal and clone the repository:

```
git clone https://github.com/YourUsername/braille_hackathon.git
cd braille_hackathon
```

Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r backend/requirements.txt
```

12. How to Run
If you did not use the automated `start.bat` script, you can run the system manually:

Start the Backend:

```
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open the Frontend: Simply double‑click `frontend/index.html` in the file explorer to open it in any modern web browser.

13. Testing for Judges
To verify the system’s capabilities without a webcam or physical Braille paper:

1. Open the frontend application in your browser.  
2. Under the camera feed, select the **Upload Image** tab.  
3. Navigate to the `dataset/sample_inputs/` folder in the project directory.  
4. Upload `hello_clean.jpg` or `braille_embossed.jpg`.  
5. The system will process the image, draw bounding boxes around the identified cells, and display the translated text in the results panel.  
6. Click the speaker icon to test the Text‑to‑Speech integration.

14. Training Reproduction Steps
To verify or improve the model accuracy, you can reproduce our training pipeline using Google Colab:

1. Create a new Google Colab notebook and select a T4 GPU runtime.  
2. Install dependencies: `!pip install ultralytics kaggle`  
3. Download the dataset via the Kaggle API:

```bash
!kaggle datasets download -d shanks0465/braille-character-dataset
!unzip -q braille-character-dataset.zip -d dataset/
```

4. Copy the contents of `train.py` into a new Colab cell. Ensure `batch=8` and `cache=False` remain set to prevent RAM exhaustion.  
5. Execute the cell. Training will complete in approximately 15‑20 minutes. The output weights can be found in `runs/detect/train/weights/best.pt`.

15. Known Limitations and Future Improvements
Shadow Directionality Bias: Extreme variations in lighting direction (e.g., lighting from the bottom edge instead of the top) can still invert dot perception. Future iterations will implement automated image rotation based on localized shadow gradient mapping prior to inference.  
Double‑Sided Interference: Standard embossed Braille paper often has dots punched on both sides. The reverse side creates “concave” dots that appear darker. The current model occasionally attempts to parse these concave dots as valid cells.  
Mobile Native Application: The current implementation relies on a web browser. Migrating the model to ONNX or CoreML for on‑device inference via a native iOS/Android application would significantly reduce latency and eliminate the need for internet connectivity.

16. Disclosure
Datasets: Models were trained utilizing open‑source community data, specifically leveraging the Braille Character Dataset hosted on Kaggle (shanks0465).  
Architecture: The core object detection utilizes the Ultralytics YOLOv8 architecture.  
Generative AI: Generative language models were utilized to assist in writing boilerplate CSS, scaffolding FastAPI endpoints, and drafting sections of this documentation. All core algorithmic logic, deployment, and model tuning were implemented manually.

17. License
This project is licensed under the MIT License – see the `LICENSE` file for details.
