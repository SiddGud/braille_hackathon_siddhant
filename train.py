from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def main():
    print("=" * 60)
    print("  BrailleVision - Optimized YOLO Retraining Script")
    print("=" * 60)
    print("\nThis script is configured to prevent RAM crashes and fix the")
    print("left/right shadow mirroring bug by disabling flip augmentations.\n")

        dataset_yaml = os.path.abspath("dataset/data.yaml")
    
    if not os.path.exists(dataset_yaml):
        print("[!] Warning: Could not find dataset/data.yaml")
        print("If you are running on Colab, you need to download the Roboflow dataset first:")
        print("  from roboflow import Roboflow")
        print("  rf = Roboflow(api_key=\"YOUR_KEY\")")
        print("  project = rf.workspace().project(\"braille-character-detection\")")
        print("  dataset = project.version(1).download(\"yolov8\")")
        return

    print(f"[*] Found dataset config at: {dataset_yaml}")

        print("[*] Loading yolov8n.pt (Nano model) to prevent OOM errors...")
    model = YOLO("yolov8n.pt")

        print("[*] Starting training...")
    print("    - Batch size: 8 (Low RAM)")
    print("    - Cache: False (Do not load images into RAM)")
    print("    - fliplr=0.0 (Crucial: Prevents w <-> r confusion)")
    print("    - flipud=0.0 (Crucial: Prevents a <-> e confusion)")
    
    results = model.train(
        data=dataset_yaml,
        epochs=50,             # Good balance of speed and accuracy
        imgsz=640,
        batch=8,               # Small batch size prevents RAM crashes
        cache=False,           # Forces disk reads instead of RAM caching
        workers=2,             # Lower workers to save memory
        name="braille_v2",
        
                augment=True,
        fliplr=0.0,            # NEVER horizontally flip Braille
        flipud=0.0,            # NEVER vertically flip Braille
        degrees=5.0,           # Only allow tiny 5-degree rotations
        hsv_h=0.015,           # Slight color jitter
        hsv_s=0.3,
        hsv_v=0.4,             # Brightness variation for shadows
        
        plots=True,
        verbose=True
    )
    
        best_pt_src = Path("runs/detect/braille_v2/weights/best.pt")
    best_pt_dst = Path("backend/models/best_v2.pt")
    
    if best_pt_src.exists():
        best_pt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt_src, best_pt_dst)
        print(f"\n[SUCCESS] New model saved to {best_pt_dst}")
        print("Update inference.py to use 'best_v2.pt' to test it!")
    else:
        print(f"\n[ERROR] Training finished but {best_pt_src} was not found.")

if __name__ == "__main__":
    main()
