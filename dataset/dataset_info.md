# Model & Dataset Information

## Model Used
- **Model Name:** `yolov8_braille` (pre-trained)
- **Source:** [DotNeuralNet — GitHub](https://github.com/snoop2head/DotNeuralNet)
- **Format:** PyTorch `.pt`
- **File:** `model/best.pt` (~52MB)
- **Task:** Object Detection — Braille character classification (a–z)
- **Classes:** 26 Braille characters

> This project integrates the publicly available pre-trained `yolov8_braille.pt` model from the DotNeuralNet open-source project.
> The primary engineering contribution of **BrailleVision** is the full-stack application built on top:
> real-time WebRTC camera streaming, WebSocket-based inference pipeline, multi-pass heuristic text repair, TTS audio output, and a dual-accessibility UI (visual dashboard + voice commands).

## Dataset (Used to train the pre-trained model)
- **Dataset Source:** Roboflow Universe — Public Braille Detection Dataset
- **Original Project:** [github.com/snoop2head/DotNeuralNet](https://github.com/snoop2head/DotNeuralNet)
- **Annotation Format:** YOLO format (`.txt` files with normalized bounding box coordinates)
- **Class Names:** `['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']`
- **Total Classes:** 26

## Note on Training
No custom model training was performed. The pre-trained `yolov8_braille` model was integrated directly.
See `training/braille_train.ipynb` for a reproducibility notebook showing how a similar model could be retrained from scratch.
