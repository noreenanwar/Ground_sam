#!/usr/bin/env python
"""
SAM wrapper module that includes conversion from predicted soft masks
to polygon embeddings. If no valid contour is found in the mask (or if the contour
area is too small), the code falls back to using the mask’s bounding rectangle
as the polygon.
"""

import cv2
import numpy as np
import torch
from torch import nn

# Import the SAM model components from the official Segment Anything library.
from segment_anything import sam_model_registry, SamPredictor

class SAMWrapper(nn.Module):
    def __init__(self, sam_checkpoint, model_type="vit_l", device="cuda", debug=False):
        """
        Initialize the SAM wrapper.
        :param sam_checkpoint: Path to the SAM checkpoint.
        :param model_type: Model type identifier (e.g., "vit_l").
        :param device: Device to run SAM on.
        :param debug: If True, print debug statements.
        """
        super().__init__()
        self.device = device
        self.debug = debug

        if self.debug:
            print("[DEBUG] Loading SAM model of type", model_type, "from checkpoint", sam_checkpoint)
        # Load the SAM model from the official registry.
        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device)
        self.model.eval()

        # Initialize the SAM predictor.
        self.predictor = SamPredictor(self.model)

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return super().to(device)

    def set_image(self, image):
        """
        Set the input image for SAM. The image is assumed to be a NumPy array
        of shape (H, W, 3) in RGB order.
        """
        self.image = image
        # Set the image for the SAM predictor.
        self.predictor.set_image(image)

    def predict_masks(self, boxes):
        """
        Given a NumPy array of bounding boxes (in pixel coordinates),
        predict segmentation masks.
        :param boxes: A NumPy array of shape (N, 4) containing bounding boxes [x1, y1, x2, y2].
        :return: A numpy array of masks.
        """
        if self.model is None:
            raise ValueError("SAM model not loaded. Please initialize SAM properly.")
        boxes_np = np.array(boxes, dtype=np.float32)
        # The official API uses the keyword argument 'box'
        masks, _, _ = self.predictor.predict(box=boxes_np, multimask_output=False)
        return masks

    def clamp_polygon(self, polygon):
        """
        Clamp the polygon coordinates to be within the image boundaries.
        The polygon is assumed to be a 1D NumPy array with 8 values in the order
        [x1, y1, x2, y2, x3, y3, x4, y4].
        :param polygon: NumPy array of shape (8,)
        :return: The clamped polygon.
        """
        if hasattr(self, "image"):
            H, W = self.image.shape[:2]
            polygon[0::2] = np.clip(polygon[0::2], 0, W)
            polygon[1::2] = np.clip(polygon[1::2], 0, H)
        return polygon

    def convert_mask_to_polygon(self, mask, threshold=0.5, min_area=10):
        mask_uint8 = (mask * 255).astype(np.uint8)
        if self.debug:
            print("[DEBUG] mask_uint8 stats: min={}, max={}".format(mask_uint8.min(), mask_uint8.max()))
        _, binary_mask = cv2.threshold(mask_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
        if self.debug:
            print("[DEBUG] Applied threshold ({}): unique values: {}".format(threshold, np.unique(binary_mask)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.debug:
            print("[DEBUG] Number of contours found:", len(contours))
        if len(contours) == 0:
            if self.debug:
                print("[DEBUG] No contours found, using boundingRect as fallback.")
            x, y, w, h = cv2.boundingRect(binary_mask)
            polygon = np.array([x, y, x+w, y, x+w, y+h, x, y+h], dtype=np.float32)
            polygon = self.clamp_polygon(polygon)
            return polygon

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if self.debug:
            print("[DEBUG] Largest contour area:", area)
        if area < min_area:
            if self.debug:
                print("[DEBUG] Contour area too small (min_area={}), using boundingRect as fallback.".format(min_area))
            x, y, w, h = cv2.boundingRect(binary_mask)
            polygon = np.array([x, y, x+w, y, x+w, y+h, x, y+h], dtype=np.float32)
            polygon = self.clamp_polygon(polygon)
            return polygon

        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygon = approx.reshape(-1)
        if polygon.shape[0] != 8:
            if self.debug:
                print("[DEBUG] Approximated polygon has {} values; using boundingRect fallback.".format(polygon.shape[0]))
            x, y, w, h = cv2.boundingRect(binary_mask)
            polygon = np.array([x, y, x+w, y, x+w, y+h, x, y+h], dtype=np.float32)
        polygon = self.clamp_polygon(polygon)
        if self.debug:
            print("[DEBUG] Final polygon points (8 values):", polygon)
        return polygon

    def convert_masks_to_polygons(self, masks, threshold=0.5, min_area=10):
        polygons = []
        for mask in masks:
            poly = self.convert_mask_to_polygon(mask, threshold=threshold, min_area=min_area)
            if poly is not None:
                poly_tensor = torch.tensor(poly, dtype=torch.float32, device=self.device)
                polygons.append(poly_tensor)
        if len(polygons) == 0:
            if self.debug:
                print("[DEBUG] No polygon embeddings generated; returning empty tensor.")
            return torch.empty((0, 8), device=self.device)
        else:
            return torch.stack(polygons, dim=0)
