import os
import zipfile
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from src.defect_detector import DefectDetector
from src.visualizer import save_masked_image
import numpy as np

# Load environment variables
load_dotenv()

def unzip_dataset(zip_path, extract_to):
    """Unzips a dataset to the specified directory."""
    print(f"Unzipping {zip_path} to {extract_to}...")
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzip complete.")
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")

def process_image(detector, image_path, output_base_dir, defect_types, dataset_label=None):
    """
    Runs detection for specified defect types on a single image.
    Returns a list of result dictionaries.
    """
    results = []
    
    # Check if image path is valid
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return results

    # Determine dataset name
    if dataset_label:
        dataset_name = dataset_label
    else:
        dataset_name = Path(image_path).parent.name
        
    image_name = Path(image_path).name
    base_name = os.path.splitext(image_name)[0]

    for defect_type in defect_types:
        try:
            # Create output directory
            output_dir = os.path.join(output_base_dir, dataset_name, defect_type)
            os.makedirs(output_dir, exist_ok=True)
            
            output_image_path = os.path.join(output_dir, f"{base_name}_result.png")
            
            # Skip if already processed
            if os.path.exists(output_image_path):
                print(f"Skipping {image_name} for {defect_type}, already processed.")
                continue
            
            # Run detection
            image, masks = detector.detect(
                str(image_path),
                defect_type=defect_type
            )
            
            # Save visualization
            output_image_path = os.path.join(output_dir, f"{base_name}_result.png")
            save_masked_image(image, masks, output_image_path, title=f"{dataset_name} - {defect_type}: {len(masks)}")
            
            # Calculate stats
            total_area = image.shape[0] * image.shape[1]
            if len(masks) > 0:
                combined_mask = np.zeros(image.shape[:2], dtype=bool)
                for m in masks:
                    combined_mask = np.logical_or(combined_mask, m['segmentation'])
                defect_area = np.sum(combined_mask)
            else:
                defect_area = 0
                
            defect_percentage = (defect_area / total_area) * 100
            
            # Save stats to list
            results.append({
                "dataset": dataset_name,
                "image": image_name,
                "defect_type": defect_type,
                "mask_count": len(masks),
                "defect_area_px": defect_area,
                "defect_percentage": defect_percentage,
                "visualization_path": output_image_path
            })
            
        except Exception as e:
            print(f"Error processing {image_path} for {defect_type}: {e}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch process defect detection.")
    parser.add_argument("--datasets", nargs='+', default=["Hires.zip", "lowres.zip"], help="List of zip files or directories.")
    parser.add_argument("--output_dir", type=str, default="results_batch", help="Directory to save results.")
    parser.add_argument("--defect_types", nargs='+', default=["crack", "biofouling", "rust"], help="Defect types to detect.")
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        model_cfg = os.getenv("MODEL_CFG", "sam2_hiera_t.yaml")
        checkpoint = os.getenv("CHECKPOINT_PATH", "checkpoints/sam2_hiera_tiny.pt")
        print(f"Initializing detector with config: {model_cfg}, checkpoint: {checkpoint}")
        detector = DefectDetector(model_cfg, checkpoint)
    except Exception as e:
        import sys
        print(f"Failed to initialize detector: {e}")
        sys.exit(1)
        
    all_results = []
    
    data_root = "data"
    os.makedirs(data_root, exist_ok=True)

    for dataset in args.datasets:
        dataset_path = Path(dataset)
        
        # Unzip if it's a zip file
        current_search_path = None
        
        if dataset_path.suffix.lower() == ".zip":
            if dataset_path.exists():
                # Extract to a subfolder named after the zip
                dataset_stem = dataset_path.stem
                extract_path = os.path.join(data_root, dataset_stem)
                
                unzip_dataset(str(dataset_path), extract_path)
                current_search_path = Path(extract_path)
            else:
                print(f"Dataset {dataset} not found.")
                continue
        elif dataset_path.is_dir():
            current_search_path = dataset_path
        else:
            print(f"Dataset source {dataset} not valid.")
            continue
            
        if current_search_path:
            print(f"Searching for images in {current_search_path}...")
            
            # Find all images recursively
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            images = []
            for ext in image_extensions:
                 images.extend(list(current_search_path.rglob(f"*{ext}")))
                 images.extend(list(current_search_path.rglob(f"*{ext.upper()}")))

            # Filter out duplicates if any (rglob might double count if case insensitive filesystem)
            images = list(set(images))
            
            print(f"Found {len(images)} images in {current_search_path}")
            
            for img_path in tqdm(images, desc=f"Processing {dataset_path.name}"):
                 # Pass the dataset name (e.g. Hires) explicitly
                 image_results = process_image(detector, str(img_path), args.output_dir, args.defect_types, dataset_label=dataset_path.stem)
                 all_results.extend(image_results)

    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, "summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Batch processing complete. Summary saved to {summary_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
