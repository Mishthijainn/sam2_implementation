import argparse
import os
import torch
from dotenv import load_dotenv
from src.defect_detector import DefectDetector
from src.visualizer import save_masked_image

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run defect detection using SAM2.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--defect_type", type=str, default="crack", choices=["crack", "biofouling", "rust"], help="Type of defect to detect.")
    parser.add_argument("--model_cfg", type=str, default=os.getenv("MODEL_CFG", "sam2_hiera_t.yaml"), help="Path to model config.")
    parser.add_argument("--checkpoint", type=str, default=os.getenv("CHECKPOINT_PATH", "checkpoints/sam2_hiera_tiny.pt"), help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "results"), help="Directory to save results.")
    
    # SAM2 Parameters
    parser.add_argument("--points_per_side", type=int, default=int(os.getenv("POINTS_PER_SIDE", 32)), help="Points per side.")
    parser.add_argument("--pred_iou_thresh", type=float, default=float(os.getenv("PRED_IOU_THRESH", 0.7)), help="Prediction IoU threshold.")
    parser.add_argument("--stability_score_thresh", type=float, default=float(os.getenv("STABILITY_SCORE_THRESH", 0.85)), help="Stability score threshold.")
    parser.add_argument("--min_mask_region_area", type=float, default=float(os.getenv("MIN_MASK_REGION_AREA", 100)), help="Minimum mask area.")
    parser.add_argument("--crop_n_layers", type=int, default=int(os.getenv("CROP_N_LAYERS", 1)), help="Crop n layers.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize detector
    try:
        detector = DefectDetector(args.model_cfg, args.checkpoint)
    except Exception as e:
        import traceback
        import sys
        print(traceback.format_exc())
        print(f"Failed to initialize DefectDetector: {e}")
        sys.exit(1)

    # Run detection
    try:
        image, masks = detector.detect(
            args.image,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_region_area,
            crop_n_layers=args.crop_n_layers,
            defect_type=args.defect_type
        )
    except Exception as e:
        print(f"Detection failed: {e}")
        return

    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_image_path = os.path.join(args.output_dir, f"{base_name}_{args.defect_type}_result.png")
    stats_path = os.path.join(args.output_dir, f"{base_name}_{args.defect_type}_stats.txt")

    # Save Results
    print(f"Saving visualization to: {output_image_path}")
    save_masked_image(image, masks, output_image_path, title=f"Detected {args.defect_type.capitalize()}s: {len(masks)}")

    # Calculate and save statistics
    total_area = image.shape[0] * image.shape[1]
    
    if len(masks) > 0:
        import numpy as np
        # Create a blank mask of the same size as the image
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for m in masks:
            combined_mask = np.logical_or(combined_mask, m['segmentation'])
        defect_area = np.sum(combined_mask)
    else:
        defect_area = 0
        
    defect_percentage = (defect_area / total_area) * 100
    
    stats = (
        f"Image: {args.image}\n"
        f"Defect Type: {args.defect_type}\n"
        f"Total Masks: {len(masks)}\n"
        f"Total Defect Area: {defect_area}\n"
        f"Defect Percentage: {defect_percentage:.2f}%\n"
        f"Parameters:\n"
        f"  pred_iou_thresh: {args.pred_iou_thresh}\n"
        f"  stability_score_thresh: {args.stability_score_thresh}\n"
        f"  min_mask_region_area: {args.min_mask_region_area}\n"
    )
    
    print("-" * 20)
    print(stats)
    print("-" * 20)
    
    with open(stats_path, "w") as f:
        f.write(stats)
    print(f"Saved stats to: {stats_path}")

if __name__ == "__main__":
    main()
