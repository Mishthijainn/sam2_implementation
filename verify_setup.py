import torch
import sys
from pathlib import Path

def check_sam2_install():
    print("Checking PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")

    print("\nChecking SAM2...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("SAM2 imports successful.")
    except ImportError as e:
        print(f"SAM2 import failed: {e}")
        return False

    print("\nChecking Assets...")
    checkpoint = Path("checkpoints/sam2.1_hiera_tiny.pt")
    if checkpoint.exists():
        print(f"Checkpoint found: {checkpoint} ({checkpoint.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"Checkpoint NOT found: {checkpoint}")
    
    config = Path("segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
    if config.exists():
         print(f"Config found: {config}")
    else:
         print(f"Config NOT found: {config}")

    return True

if __name__ == "__main__":
    if check_sam2_install():
        print("\nVerification Passed!")
    else:
        print("\nVerification Failed!")
