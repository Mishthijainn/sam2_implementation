import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

def build_sam2_manual(config_file, ckpt_path, device):
    try:
        cfg = OmegaConf.load(config_file)
        model = instantiate(cfg.model, _recursive_=True)
        
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
            model.load_state_dict(sd)
            
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Manual build failed: {e}")
        raise

class DefectDetector:
    def __init__(self, model_cfg, checkpoint_path, device=None):
        """
        Initialize the DefectDetector with SAM2 model.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading SAM2 model from {checkpoint_path} on {self.device}...")
        
        try:
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        except Exception as e:
            print(f"Standard build_sam2 failed: {e}. Trying manual load...")
            self.sam2_model = build_sam2_manual(model_cfg, checkpoint_path, self.device)
            
    def detect(self, image_path, 
               points_per_side=32, 
               pred_iou_thresh=0.7, 
               stability_score_thresh=0.85, 
               crop_n_layers=1, 
               crop_n_points_downscale_factor=2, 
               min_mask_region_area=100,
               defect_type="crack"):
        """
        Detect defects in an image using SAM2.
        
        Args:
            image_path (str): Path to the input image.
            points_per_side (int): Number of points to sample along one side of the image.
            pred_iou_thresh (float): Filtering threshold for predicted mask IoU.
            stability_score_thresh (float): Filtering threshold for stability score.
            crop_n_layers (int): Number of layers to crop the image.
            crop_n_points_downscale_factor (int): Downscale factor for points in crop layers.
            min_mask_region_area (int): Minimum area of mask region to keep.
            defect_type (str): Type of defect to detect ("crack", "biofouling", "rust").
            
        Returns:
            tuple: (image (np.ndarray), masks (list of dicts))
        """
        
    
        if defect_type == "crack":
             pass
        elif defect_type == "biofouling":
            if stability_score_thresh > 0.8:
                stability_score_thresh = 0.8 # Relax slightly
        elif defect_type == "rust":
            pass

      
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
        
        # Load and prepare image
        try:
            image_pil = Image.open(image_path)
            
            max_size = 1024
            if max(image_pil.size) > max_size:
                ratio = max_size / max(image_pil.size)
                new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
                print(f"Resizing image from {image_pil.size} to {new_size} for SAM2 processing.")
                image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
                
            image = np.array(image_pil.convert("RGB"))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        print(f"Processing image: {image_path} (Shape: {image.shape}) for defect type: {defect_type}")
        print("Generating masks...")
        
        try:
            raw_masks = self.mask_generator.generate(image)
        except Exception as e:
            print(f"Error during mask generation: {e}")
            raise

        # Filter masks based on defect type
        filtered_masks = self.filter_masks(raw_masks, defect_type, image)
        print(f"Generated {len(raw_masks)} raw masks. Filtered down to {len(filtered_masks)} {defect_type} masks.")

        return image, filtered_masks

    def filter_masks(self, masks, defect_type, image):
        """
        Filter masks depending on the defect type, using shape and color heuristics.
        """
        filtered = []
        for mask in masks:
            area = mask['area']
            bbox = mask['bbox'] # [x, y, w, h]
            w, h = bbox[2], bbox[3]
            segmentation = mask['segmentation']
            
            # Prevent division by zero
            if w == 0 or h == 0:
                continue
                
            aspect_ratio = max(w, h) / min(w, h)
            
            # Extract color information for the mask
            # get pixels where mask is true
            masked_pixels = image[segmentation]
            if len(masked_pixels) > 0:
                mean_r = np.mean(masked_pixels[:, 0])
                mean_g = np.mean(masked_pixels[:, 1])
                mean_b = np.mean(masked_pixels[:, 2])
            else:
                mean_r = mean_g = mean_b = 0
            
            if defect_type == "crack":
                # Cracks are typically thin and long
                # Heuristic: High aspect ratio
                if aspect_ratio > 2.0: 
                   filtered.append(mask)
            elif defect_type == "biofouling":
                # Biofouling: Organic, blob-like, lower aspect ratio
                # Color heuristic: Look for green or dark pixels
                brightness = (mean_r + mean_g + mean_b) / 3
                if aspect_ratio < 3.0:
                    is_dark = brightness < 100  # Threshold for "dark"
                    is_green = mean_g > mean_r and mean_g > mean_b * 0.9  # Threshold for "green"
                    
                    if is_dark or is_green:
                        filtered.append(mask)
            elif defect_type == "rust":
                # Rust: Patches
                # Color heuristic: Predominantly reddish/brown/orange
                # Simple check: R > G and R > B by some margin
                if aspect_ratio < 3.0:
                    if mean_r > mean_g * 1.05 and mean_r > mean_b * 1.05:
                        filtered.append(mask)
            else:
                # Default: no specific filtering
                filtered.append(mask)
                
        return filtered
