import os
import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image

# Import our wrapper (assuming running from src/ or adjusting path)
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Utils
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


# ... (imports)
import json

# ... (previous code)

def run_experiment(image_path, model_cfg, checkpoint, output_dir):
    image_name = Path(image_path).stem
    exp_dir = output_dir / image_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running experiments on {image_name}...")
    stats = {"image": image_name}

    # Load Image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    h, w, _ = image.shape
    
    # Init SAM2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    
    # --- Experiment 1: Single Point Prompt (Center) ---
    input_point = np.array([[w//2, h//2]])
    input_label = np.array([1]) 
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    stats["exp1_score"] = float(scores[best_idx])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[best_idx], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Exp 1: Single Point (Score: {scores[best_idx]:.3f})")
    plt.axis('off')
    plt.savefig(exp_dir / "exp1_single_point.png")
    plt.close()

    # --- Experiment 2: Multi-point (Fg + Bg) ---
    input_point_multi = np.array([[w//2, h//2], [w//10, h//10]])
    input_label_multi = np.array([1, 0]) 
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point_multi,
        point_labels=input_label_multi,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    stats["exp2_score"] = float(scores[best_idx])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[best_idx], plt.gca())
    show_points(input_point_multi, input_label_multi, plt.gca())
    plt.title(f"Exp 2: Multi Point Fg+Bg (Score: {scores[best_idx]:.3f})")
    plt.axis('off')
    plt.savefig(exp_dir / "exp2_multi_point.png")
    plt.close()
    
    # --- Experiment 3: Bounding Box ---
    box = np.array([w//4, h//4, 3*w//4, 3*h//4])
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False, 
    )
    stats["exp3_score"] = float(scores[0])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(box, plt.gca())
    plt.title(f"Exp 3: Box Prompt (Score: {scores[0]:.3f})")
    plt.axis('off')
    plt.savefig(exp_dir / "exp3_box.png")
    plt.close()

    return stats

def main():
    base_dir = Path("sam2_project") 
    if not base_dir.exists():
        base_dir = Path(".")
        
    data_dir = base_dir / "data"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Config
    if (base_dir / "segment-anything-2").exists():
        repo_root = base_dir / "segment-anything-2"
    else:
        repo_root = base_dir.parent / "segment-anything-2" 

    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    checkpoint = base_dir / "checkpoints" / "sam2.1_hiera_tiny.pt"  
    
    if not checkpoint.exists():
        print(f"Checkpoint not found at {checkpoint}")
        return

    images = list(data_dir.glob("*.jpg"))
    if not images:
        print("No images found in data/")
        return
        
    all_stats = []
    for img in images:
        try:
            stats = run_experiment(img, model_cfg, checkpoint, output_dir)
            all_stats.append(stats)
        except Exception as e:
            print(f"Failed to run on {img}: {e}")
            with open("error.log", "a") as f:
                f.write(f"Error processing {img}:\n")
                traceback.print_exc(file=f)
                f.write("\n" + "-"*40 + "\n")
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved metrics to {output_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
