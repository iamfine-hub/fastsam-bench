"""
Local reasoning test script for FastSAM.
Tests the model on a single image — everything mode (segment all objects).
"""
import os
import sys
import torch
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt

# ---- config ----
IMAGE_PATH = "/root/FastSAM_clone/test_img/white_car.png"
MODEL_PATH = "/root/FastSAM_clone/FastSAM-x.pt"   # auto-downloads from ultralytics if not present
OUTPUT_PATH = "/root/FastSAM_clone/test_img/output_white_car.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Device: {DEVICE}")
    print(f"Image : {IMAGE_PATH}")
    print(f"Model : {MODEL_PATH}")

    # 1. Load model
    print("\n[1/3] Loading model...")
    model = FastSAM(MODEL_PATH)
    print("       Done.")

    # 2. Run inference (everything mode)
    print("[2/3] Running inference...")
    everything_results = model(
        IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    if everything_results is None or len(everything_results) == 0:
        print("       No objects detected.")
        sys.exit(1)
    print(f"       Detected {len(everything_results[0].boxes)} objects.")

    # 3. Render and save result
    print("[3/3] Saving result...")
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
    ann = prompt_process.everything_prompt()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    prompt_process.plot(
        annotations=ann,
        output_path=OUTPUT_PATH,
        mask_random_color=True,
        retina=True,
        withContours=True,
    )
    print(f"       Saved to {OUTPUT_PATH}")
    print("\nTest passed — FastSAM is working.")

if __name__ == "__main__":
    main()
