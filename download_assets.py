import os
import requests
from pathlib import Path

BASE_DIR = Path(".")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"

CHECKPOINTS = {
    # "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    # "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    # "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
}

# Using a few fixed sample images from COCO (using val2017 links)
COCO_IMAGES = {
    "COCO_val2017_000000039769.jpg": "http://images.cocodataset.org/val2017/000000039769.jpg", # Cats (Standard)
    "COCO_val2017_000000000139.jpg": "http://images.cocodataset.org/val2017/000000000139.jpg", # Living room (Complex/Cluttered)
    "COCO_val2017_000000000285.jpg": "http://images.cocodataset.org/val2017/000000000285.jpg", # Bear (Single object)
    "COCO_val2017_000000000632.jpg": "http://images.cocodataset.org/val2017/000000000632.jpg", # Bedroom (Interior/Perspective)
}

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    print("Downloading SAM2 Checkpoints...")
    for name, url in CHECKPOINTS.items():
        download_file(url, CHECKPOINTS_DIR / name)

    print("\nDownloading Sample Images...")
    for name, url in COCO_IMAGES.items():
        download_file(url, DATA_DIR / name)

if __name__ == "__main__":
    main()
