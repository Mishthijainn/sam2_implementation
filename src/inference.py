import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Wrapper:
    def __init__(self, checkpoint_path, model_cfg, device="cuda"):
        # Select device automatically if not specified or available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = "cpu"
        
        self.device = device
        print(f"Loading SAM2 model from {checkpoint_path} on {self.device}...")
        self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.current_image = None

    def set_image(self, image_path):
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        self.current_image = image
        self.predictor.set_image(image)
        return image

    def predict_point(self, point_coords, point_labels):
        """
        point_coords: numpy array of shape (N, 2)
        point_labels: numpy array of shape (N,), 1 for fg, 0 for bg
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Sort by score
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits

    def predict_box(self, box):
        """
        box: numpy array of shape (4,) [x1, y1, x2, y2]
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits

    def predict_box_and_point(self, box, point_coords, point_labels):
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box[None, :],
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits
