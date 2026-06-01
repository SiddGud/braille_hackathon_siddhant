# Dataset Information

> **NOTE TO JUDGES:** The full dataset (~60MB) was sourced from Roboflow Universe and used for training on Google Colab (T4 GPU). If you need access, please use the download link below or contact us for temporary access.

### Dataset Details
- **Dataset Source:** Roboflow Universe — Public Braille Detection Dataset (~60MB)
- **Dataset Link:** `[INSERT YOUR ROBOFLOW PROJECT LINK HERE — check your Colab notebook for the exact workspace/project name]`
- **Annotation Format:** YOLO format (`.txt` files with normalized bounding box coordinates)
- **Class Names:** `['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']`
- **Total Classes:** 26
- **Train/Val/Test Split:** `[INSERT — check your Roboflow project dashboard]`
- **Total Images:** `[INSERT — check your Roboflow project dashboard]`

### Training Environment
- **Platform:** Google Colab (T4 GPU)
- **Base Model:** YOLOv8n (yolov8n.pt — nano variant)
- **Training Notebook:** `training/braille_train.ipynb`

### Preprocessing Steps
- YOLO-format annotations with normalized bounding box coordinates
- Images augmented with rotation (±5°), brightness shift, mosaic, and mixup during training
- `fliplr=0.0` (flip disabled — Braille patterns are direction-sensitive)
