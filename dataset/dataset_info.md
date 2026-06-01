# Model & Dataset Information

## Model Used
- **Type:** Pre-trained YOLOv8 Braille Detection Model
- **Source:** Downloaded from Roboflow Universe (public pre-trained model)
- **Format:** PyTorch `.pt`
- **File:** `model/best.pt` (~52MB)
- **Classes:** 26 Braille characters (a–z)

> This project uses an existing publicly available pre-trained YOLOv8 model for Braille character detection.
> The core contribution of this project is the **full-stack application layer** built on top of the model:
> real-time WebRTC streaming, WebSocket-based inference pipeline, heuristic text repair algorithms, and TTS output.

## Dataset (Used by the Pre-trained Model)
- **Dataset Source:** Roboflow Universe — Public Braille Detection Dataset
- **Dataset Link:** `[INSERT ROBOFLOW PROJECT LINK — check where you downloaded the model from]`
- **Annotation Format:** YOLO format
- **Class Names:** `['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']`

## Note on Training
No custom training was performed. The pre-trained model was integrated as-is into the BrailleVision inference pipeline. All custom engineering work was done at the application layer.

See `training/braille_train.ipynb` for a reference training notebook provided for reproducibility (shows how a similar model could be retrained from scratch using Roboflow data).
