# Dataset Information

> **NOTE TO JUDGES:** If the dataset is too large to host directly in this GitHub repository, please use the secure download link provided below to access the full `images/` and `labels/` directories required to reproduce our YOLO training.

### Dataset Details
- **Dataset Link:** `[INSERT LINK TO YOUR DATASET HERE]`
- **Dataset Source:** Custom physical Braille captures
- **Annotation Format:** YOLO format (`.txt` files mapping class IDs and normalized bounding box coordinates)
- **Class Names:** `['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']`
- **Total Images:** `[INSERT TOTAL IMAGES]`
- **Train/Val/Test Split:** `[INSERT SPLIT PERCENTAGES, e.g. 80/10/10]`

### Preprocessing Steps
- Images were captured using physical indented Braille paper (back of the page).
- A standardized bottom-left low-angle light source was used to create consistent shadow maps for the computer vision model.
- Images were resized to `1280x720` maintaining aspect ratio.
