# Model & Dataset Information

## Model Used
- **Base Model:** YOLOv8n (Nano) from Ultralytics
- **Fine-Tuning:** Fine-tuned for 50 epochs on Braille-specific datasets
- **Format:** PyTorch `.pt`
- **File:** `model/best.pt` (~52 MB)
- **Task:** Object Detection — Braille character classification
- **Classes:** 26 (a-z)
- **Final Metrics:** mAP50 of 0.989 on the validation set
- **Training Hardware:** NVIDIA T4 Tensor Core GPU (Google Colab)

> The YOLOv8n base model was fine-tuned on a combination of the Angelina Braille Dataset and the DSBI double-sided dataset. On top of the model we built a custom full-stack application layer: HTTP-based inference pipeline, multi-pass heuristic text repair, TTS audio output, and a dual-accessibility UI.

## Dataset Information
- **Primary Dataset:** Angelina Braille Dataset (Roboflow Universe)
- **Secondary Dataset:** DSBI Double-Sided Braille Images (Academic Repository)
- **Total Volume:** ~1,800 images
- **Annotation Format:** YOLO format (`.txt` files with normalized bounding box coordinates)
- **Class Names:** a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z (26 classes)
- **Dataset Download:** Full dataset can be downloaded via Roboflow API (see Training Reproduction section in README)

## Train / Validation / Test Split
- **Train:** ~1,350 images (75%)
- **Validation:** ~270 images (15%)
- **Test:** ~180 images (10%)
- **Split Method:** Random stratified split performed by Roboflow during dataset export

Note: Only a representative subset of the training data is included directly in the repository (`dataset/images/train/` and `dataset/images/val/`). The full dataset can be reproduced by downloading it via the Roboflow API as documented in the README.

## Preprocessing Steps
1. **Resize:** All images resized to 640x640 (YOLO default input size)
2. **Morphological Top-Hat:** Elliptical kernel (3x3) applied to enhance dot contrast against paper background
3. **CLAHE:** Contrast Limited Adaptive Histogram Equalization (clipLimit=3.0, tileGridSize=8x8) to normalize uneven lighting across the image
4. **Adaptive Thresholding:** Gaussian adaptive threshold (blockSize=15, C=2) to binarize dot regions
5. **Inversion:** Binary inversion so Braille dots appear as white on black background
6. **Multi-variant inference:** At runtime, three image variants are generated (original, high-contrast CLAHE, darkened gamma=2.0) and the best result is selected

## Key Training Decisions
- **fliplr=0.0, flipud=0.0:** Flip augmentations were explicitly disabled because Braille is chiral — mirroring a character changes its meaning (e.g., 'r' becomes 'w')
- **Batch Size:** 8 (optimized for Colab T4 memory constraints)
- **cache=False:** Disabled dataset caching to prevent RAM exhaustion on free-tier Colab

## Training Reproduction
The training notebook (`training/braille_train.ipynb`) contains the full training pipeline. The `best.pt` checkpoint is the output of this training process and is provided directly in the repository.
