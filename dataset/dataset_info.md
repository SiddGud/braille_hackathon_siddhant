# Model & Dataset Information

## Model Used
- **Base Model:** `yolov8_braille` (Pre-trained DotNeuralNet model)
- **Fine-Tuning:** No custom fine-tuning was performed. The pre-trained model is used directly.
- **Format:** PyTorch `.pt`
- **File:** `model/best.pt` (~52 MB) (Also available via Google Drive link)
- **Task:** Object Detection — Braille character classification
- **Classes:** 64 binary dot pattern classes

> This project utilizes the pre‑trained `yolov8_braille` model from the DotNeuralNet repository. On top of the model we built a custom full‑stack application layer:
> WebSocket‑based inference pipeline, multi‑pass heuristic text repair, TTS audio output, and a dual‑accessibility UI.

## Dataset Information
- **Dataset Origin:** Public Braille Detection datasets (Roboflow Universe)
- **Annotation Format:** YOLO format (`.txt` files with normalized bounding box coordinates)

## Note on Training
The training notebook (`training/braille_train.ipynb`) demonstrates how the pre‑trained model can be fine‑tuned, but for this submission we used the original weights without additional training. The `best.pt` checkpoint is provided directly in the repository.
