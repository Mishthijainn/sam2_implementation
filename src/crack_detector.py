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
    # Manual loading to bypass Hydra search path issues
    try:
        cfg = OmegaConf.load(config_file)
        model = instantiate(cfg.model, _recursive_=True)
        
        if ckpt_path is not None:
            # Load checkpoint
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
            model.load_state_dict(sd)
            
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Manual build failed: {e}")
        raise

class CrackDetector:
    def __init__(self, model_cfg, checkpoint_path, device=None):
        """
        Initialize the CrackDetector with SAM2 model.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading SAM2 model from {checkpoint_path} on {self.device}...")
        
        try:
            # Try standard build first
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        except Exception as e:
            print(f"Standard build_sam2 failed: {e}. Trying manual load...")
            # Fallback to manual loading
            # If model_cfg is just a name, and not found, we might need absolute path.
            # But the caller should ideally provide valid path.
            self.sam2_model = build_sam2_manual(model_cfg, checkpoint_path, self.device)
            
        self.mask_generator = None

    def detect(self, image_path, 
               points_per_side=32, 
               pred_iou_thresh=0.7, 
               stability_score_thresh=0.85, 
               crop_n_layers=1, 
               crop_n_points_downscale_factor=2, 
               min_mask_region_area=100):
        """
        Detect cracks in an image using SAM2.
        
        Args:
            image_path (str): Path to the input image.
            points_per_side (int): Number of points to sample along one side of the image.
            pred_iou_thresh (float): Filtering threshold for predicted mask IoU.
            stability_score_thresh (float): Filtering threshold for stability score.
            crop_n_layers (int): Number of layers to crop the image.
            crop_n_points_downscale_factor (int): Downscale factor for points in crop layers.
            min_mask_region_area (int): Minimum area of mask region to keep.
            
        Returns:
            tuple: (image (np.ndarray), masks (list of dicts))
        """
        # Initialize generator with specified parameters
        # Re-initializing allows changing parameters between calls if needed
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
            image = np.array(image_pil.convert("RGB"))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        print(f"Processing image: {image_path} (Shape: {image.shape})")
        print("Generating masks...")
        
        try:
            masks = self.mask_generator.generate(image)
            print(f"Successfully generated {len(masks)} masks.")
        except Exception as e:
            print(f"Error during mask generation: {e}")
            raise

        return image, masks
