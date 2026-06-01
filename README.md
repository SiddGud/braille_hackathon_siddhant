# BrailleVision
Real-Time Optical Braille Recognition and Translation System

1. Problem Statement
According to the World Health Organization, at least 2.2 billion people globally have a near or distance vision impairment. While Braille remains the fundamental tool for literacy, education, and employment for the visually impaired, Braille literacy rates have been declining. A major contributing factor is the lack of accessible translation tools. Sighted teachers, parents, and peers often cannot read Braille, creating a significant communication barrier. Traditional Braille translation requires expensive, specialized hardware or manual transcription, which is slow and inaccessible to the general public.

2. Solution Overview
BrailleVision bridges this communication gap by democratizing Braille translation. It is an end-to-end, real-time computer vision system that translates physical, embossed Braille into digital text and audible speech using any standard web camera or smartphone camera. By leveraging state-of-the-art deep learning object detection, BrailleVision achieves high accuracy across various lighting conditions, paper types, and camera angles, making Braille instantly readable for anyone.

3. System Architecture
The system operates on a highly optimized, low-latency client-server architecture designed for real-time video stream processing.

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
[ Render UI & Trigger Text-to-Speech ]

4. Single-Stage Detection Pipeline
The core of BrailleVision is an end-to-end deep learning pipeline. Unlike traditional optical Braille recognition that struggles with uneven shadows on embossed paper, our system uses a unified object detection approach. The YOLO architecture simultaneously predicts bounding boxes and class probabilities (the 6-bit binary state of the Braille cell) directly from full images in a single evaluation. This unified architecture allows the system to process video frames in real-time while remaining highly resilient to complex lighting environments.

5. Model Details
Base Architecture: YOLOv8n (Nano)
Fine-Tuning: The model was fine-tuned for 50 epochs using a highly constrained augmentation strategy.
Crucial Hyperparameters: fliplr=0.0 and flipud=0.0. Horizontal and vertical flipping augmentations were explicitly disabled during training. Braille is chiral; flipping a character horizontally changes its fundamental meaning (e.g., 'w' and 'r' are mirrors of each other). Disabling these augmentations prevented catastrophic class confusion during training.
Batch Size: 8 (optimized for lower-tier hardware constraints).
Final Metrics: Achieved an mAP50 of 0.989 on the validation set.
Training Hardware: NVIDIA T4 Tensor Core GPU.

6. Preprocessing & Fallback Logic
When the neural network detects an unknown binary pattern (outputting a '?' due to ambiguous shadow angles), the system employs a deterministic mirror-swap fallback. By swapping the left column with the right column in the detected binary string, it mathematically corrects instances where the lighting direction caused the neural network's spatial pooling to invert the shadow perception.

7. Dataset Details
Dataset Name	Source	Purpose	Volume
Angelina Braille Dataset	Roboflow / Open Source	Primary training data. Contains complex real-world shadows and warped paper.	~1,500 images
DSBI (Double-Sided)	Academic Repository	Edge-case training for double-sided punched paper interference.	~300 images
Synthetic Generator	Custom Python Script	Baseline bootstrapping and perfect-condition validation matrices.	Dynamically Generated

8. Technology Stack
Layer	Technology
Frontend	HTML5, CSS3 (Vanilla), JavaScript, Canvas API
Backend API	Python 3.11, FastAPI, Uvicorn
Machine Learning	Ultralytics (YOLOv8), PyTorch
Audio Processing	gTTS (Google Text-to-Speech), Pygame

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
│       └── best.pt            # Fine-tuned YOLOv8 neural network weights
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
Double click start.bat. This script will automatically create a virtual environment, install all dependencies, start the backend server, and open the frontend in your default browser.

Mac / Linux (Manual)
Ensure Python 3.11+ is installed.
Open a terminal and clone the repository:
git clone https://github.com/YourUsername/braille_hackathon.git
cd braille_hackathon
Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate
Install dependencies:
pip install -r backend/requirements.txt

12. How to Run
If you did not use the automated start.bat script, you can run the system manually:

Start the Backend:
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Open the Frontend: Simply double-click frontend/index.html in the file explorer to open it in any modern web browser.

13. Testing for Judges
To verify the system's capabilities without a webcam or physical Braille paper:

Open the frontend application in your browser.
Under the camera feed, select the Upload Image tab.
Navigate to the dataset/sample_inputs/ folder in the project directory.
Upload hello_clean.jpg or braille_embossed.jpg.
The system will process the image, draw bounding boxes around the identified cells, and display the translated text in the results panel.
Click the speaker icon to test the Text-to-Speech integration.

14. Training Reproduction Steps
To verify or improve the model accuracy, you can reproduce our training pipeline using Google Colab:

Create a new Google Colab notebook and select a T4 GPU runtime.
Install dependencies: !pip install ultralytics roboflow
Download the dataset via the Roboflow API (requires a free API key):
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("braille-character-detection")
dataset = project.version(1).download("yolov8")
Copy the contents of train.py into a new Colab cell. Ensure batch=8 and cache=False remain set to prevent RAM exhaustion.
Execute the cell. Training will complete in approximately 15-20 minutes. The output weights can be found in runs/detect/train/weights/best.pt.

15. Known Limitations and Future Improvements
Shadow Directionality Bias: Extreme variations in lighting direction (e.g., lighting from the bottom edge instead of the top) can still invert dot perception. Future iterations will implement automated image rotation based on localized shadow gradient mapping prior to inference.
Double-Sided Interference: Standard embossed Braille paper often has dots punched on both sides. The reverse side creates "concave" dots that appear darker. The current model occasionally attempts to parse these concave dots as valid cells.
Mobile Native Application: The current implementation relies on a web browser. Migrating the model to ONNX or CoreML for on-device inference via a native iOS/Android application would significantly reduce latency and eliminate the need for internet connectivity.

16. Disclosure
Datasets: Models were trained utilizing open-source community data, specifically leveraging variations of the Angelina Braille Dataset hosted on Roboflow Universe.
Architecture: The core object detection utilizes the Ultralytics YOLOv8 architecture.
Generative AI: Generative language models were utilized to assist in writing boilerplate CSS, scaffolding FastAPI endpoints, and drafting sections of this documentation. All core algorithmic logic, deployment, and model tuning were implemented manually.

17. License
This project is licensed under the MIT License - see the LICENSE file for details.
