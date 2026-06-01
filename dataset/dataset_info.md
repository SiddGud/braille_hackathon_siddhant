# Model & Dataset Information

## Model Used
- **Base Model:** YOLOv8n (Nano) from Ultralytics
- **Fine-Tuning:** Fine-tuned for 50 epochs on Braille-specific datasets
- **Format:** PyTorch `.pt`
- **File:** `model/best.pt` (~52 MB)
- **Task:** Object Detection — Braille character classification
- **Classes:** 64 binary dot pattern classes
- **Final Metrics:** mAP50 of 0.989 on the validation set
- **Training Hardware:** NVIDIA T4 Tensor Core GPU (Google Colab)

> The YOLOv8n base model was fine-tuned on a combination of the Angelina Braille Dataset and the DSBI double-sided dataset. On top of the model we built a custom full-stack application layer: HTTP-based inference pipeline, multi-pass heuristic text repair, TTS audio output, and a dual-accessibility UI.

## Dataset Information
- **Dataset Origin:** Angelina Braille Dataset (Roboflow Universe) + DSBI Double-Sided Dataset
- **Total Volume:** ~1,800 images
- **Annotation Format:** YOLO format (`.txt` files with normalized bounding box coordinates)

## Key Training Decisions
- **fliplr=0.0, flipud=0.0:** Flip augmentations were explicitly disabled because Braille is chiral — mirroring a character changes its meaning (e.g., 'r' becomes 'w').
- **Batch Size:** 8 (optimized for Colab T4 memory constraints)
- **cache=False:** Disabled dataset caching to prevent RAM exhaustion on free-tier Colab

## Training Reproduction
The training notebook (`training/braille_train.ipynb`) contains the full training pipeline. The `best.pt` checkpoint is the output of this training process and is provided directly in the repository.
