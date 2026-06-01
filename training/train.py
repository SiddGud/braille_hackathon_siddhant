from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def main():
    print("=" * 70)
    print("  BrailleVision - SOTA Extreme Accuracy Training (NO CONSTRAINTS)")
    print("=" * 70)
    print("\n[+] Initializing State-of-the-Art Training Pipeline...")
    print("[+] Using YOLOv8x (Extra Large) and 1280px Resolution")

    dataset_yaml = os.path.abspath("dataset/data.yaml")
    
    if not os.path.exists(dataset_yaml):
        print("\n[!] Dataset not found locally. To download from Roboflow:")
        print("  1. Go to https://universe.roboflow.com/search?q=braille")
        print("  2. Click any good dataset, select 'YOLOv8' format, and click 'Show Download Code'")
        print("  3. Paste that Python snippet into your notebook before running this training script!")
        return

    # 1. Use the massive RT-DETR model (Vision Transformer)
    # This abandons CNNs entirely and uses self-attention (like ChatGPT) to understand global page context.
    from ultralytics import RTDETR
    print("[*] Loading rtdetr-l.pt (Transformer)...")
    model = RTDETR("rtdetr-l.pt")

    # 2. Extreme Hyperparameter Tuning for Small Object Detection (Braille)
    print("[*] Starting extreme training loop...")
    
    results = model.train(
        data=dataset_yaml,
        epochs=100,             # Deep training for maximum convergence
        imgsz=1024,             # 1024px prevents dot resolution loss while saving GPU VRAM
        batch=4,                # CRITICAL: Low batch size prevents GPU VRAM crashes
        cache=False,            # CRITICAL: False prevents system RAM crashes on free tier
        workers=2,              # CRITICAL: Lower CPU threads
        name="braille_sota_free_v1",
        
        # --- SOTA AUGMENTATION (HOW THE PROS WIN HACKATHONS) ---
        
        # The Golden Rule for Braille: Never flip.
        fliplr=0.0,             # Prevents w -> r hallucination
        flipud=0.0,             # Prevents a -> e hallucination
        
        # Geometric Warping (Simulates a user holding a phone at a weird angle)
        degrees=15.0,           # Allows up to 15 degree natural tilt
        perspective=0.001,      # Simulates 3D camera tilt (crucial for paper on a desk)
        shear=2.0,              # Simulates paper warping
        
        # Lighting & Shadow Chaos (Simulates bad room lighting)
        hsv_h=0.02,             # Hue variance
        hsv_s=0.7,              # Heavy saturation variance
        hsv_v=0.7,              # Extreme brightness/shadow variance
        
        # Compositional Augmentation (Forces the model to actually look at dots, not page layout)
        mosaic=1.0,             # Combines 4 images into 1 (standard in YOLOv8 SOTA)
        mixup=0.2,              # Overlays images to simulate extreme noise
        copy_paste=0.2,         # Copies braille cells to random spots
        
        plots=True,
        verbose=True,
        optimizer='auto'
    )
    
    best_pt_src = Path("runs/detect/braille_sota_v1/weights/best.pt")
    best_pt_dst = Path("backend/models/best_sota.pt")
    
    if best_pt_src.exists():
        best_pt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt_src, best_pt_dst)
        print(f"\n[SUCCESS] SOTA model saved to {best_pt_dst}")
    else:
        print(f"\n[ERROR] Training finished but {best_pt_src} was not found.")

if __name__ == "__main__":
    main()
