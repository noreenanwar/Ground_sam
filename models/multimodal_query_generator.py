import torch
import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor
# (No concurrent.futures import here since we’re doing sequential processing)

# Import Grounding DINO functions
from groundingdino.util.inference import (
    load_model as gdino_load_model,
    predict as gdino_predict
)

# Import SAM components
from segment_anything import SamPredictor, sam_model_registry

############################################
# Configuration and Checkpoint Paths
############################################
GDINO_CONFIG_PATH = "config/GroundingDINO_SwinT_OGC.py"       # Adjust as needed
GDINO_WEIGHTS_PATH = "checkpoints/groundingdino_swint_ogc.pth"  # Adjust as needed
SAM_CHECKPOINT = "checkpoints/sam_vit_l_0b3195.pth"            # Adjust as needed

############################################
# Helper Module: Polygon Encoder
############################################
class PolygonEncoder(nn.Module):
    """
    A simple MLP to encode polygon coordinates into a fixed-size embedding.
    Assumes a maximum of 100 points (i.e. 200 numbers).
    """
    def __init__(self, input_dim=200, hidden_dim=256, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, polygon_coords):
        return self.mlp(polygon_coords)

def get_polygon_coords_from_sam(masks):
    """
    Converts SAM-generated masks into polygon coordinates using OpenCV’s findContours.
    """
    polygons = []
    for mask in masks:
        if mask is None:
            polygons.append(None)
            continue
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and len(contours) > 0:
            polygon = contours[0].reshape(-1, 2)
        else:
            polygon = None
        polygons.append(polygon)
    return polygons

############################################
# Query Generator Class
############################################
class QueryGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', num_queries=300, hidden_dim=256):
        self.device = device
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Load and cache the heavy models just once.
        self.gdino_model = gdino_load_model(GDINO_CONFIG_PATH, GDINO_WEIGHTS_PATH).to(self.device)
        self.sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)
        self.polygon_encoder = PolygonEncoder(input_dim=200, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim).to(self.device)

    def generate(self, pil_image, text_prompt):
        """
        Generates multi-modal queries for the given PIL image and text prompt.
        
        Returns a dictionary with:
          - appearance_queries: Tensor of appearance-based queries.
          - positional_queries: Tensor of positional queries.
          - random_queries: Extra random queries.
          - combined_queries: Final concatenated & normalized query tensor.
          - boxes, phrases, masks, polygons: Additional debugging outputs.
        """
        # ----- Step 0: Prepare the image -----
        image_tensor = ToTensor()(pil_image).to(self.device)

        # ----- Step 1: Appearance Queries via Grounding DINO -----
        boxes, logits, phrases = gdino_predict(
            model=self.gdino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=0.30,
            text_threshold=0.25
        )
        # For demonstration, generate random appearance embeddings (one per detected phrase)
        appearance_queries = torch.randn(len(phrases), self.hidden_dim, device=self.device)

        # ----- Step 2: Positional Queries via SAM -----
        image_np = np.array(pil_image)
        self.sam_predictor.set_image(image_np)
        H, W, _ = image_np.shape
        # Convert boxes from Grounding DINO to pixel coordinates.
        boxes_xywh = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            boxes_xywh.append([x_min * W, y_min * H, w * W, h * H])
        boxes_xywh = np.array(boxes_xywh)

        masks = []
        for box_xywh in boxes_xywh:
            mask_output, scores, logits_mask = self.sam_predictor.predict(box=box_xywh, multimask_output=False)
            if mask_output is None or mask_output.size == 0:
                masks.append(None)
            else:
                mask = mask_output[0] if len(mask_output.shape) == 3 else mask_output
                masks.append(mask)

        polygons = get_polygon_coords_from_sam(masks)
        positional_queries_list = []
        max_points = 100
        for polygon in polygons:
            if polygon is None or len(polygon) == 0:
                coords = np.zeros(max_points * 2, dtype=np.float32)
            else:
                coords = polygon.reshape(-1)
                if coords.shape[0] >= max_points * 2:
                    coords = coords[:max_points * 2]
                else:
                    coords = np.pad(coords, (0, max_points * 2 - coords.shape[0]), 'constant', constant_values=0)
            coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
            poly_query = self.polygon_encoder(coords_tensor)
            positional_queries_list.append(poly_query)
        if len(positional_queries_list) == 0:
            positional_queries = torch.empty((0, self.hidden_dim), device=self.device)
        else:
            positional_queries = torch.stack(positional_queries_list, dim=0)

        # ----- Step 3: Extra Random Queries -----
        total_explicit = appearance_queries.shape[0] + positional_queries.shape[0]
        num_random_queries = self.num_queries - total_explicit
        if num_random_queries < 0:
            raise RuntimeError("num_queries is too small for the given explicit queries.")
        random_queries = torch.randn(num_random_queries, self.hidden_dim, device=self.device)

        # ----- Step 4: Concatenate and Normalize Queries -----
        combined_queries = torch.cat([appearance_queries, positional_queries, random_queries], dim=0)
        std = combined_queries.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-6, torch.tensor(1e-6, device=self.device), std)
        combined_queries = (combined_queries - combined_queries.mean(dim=0, keepdim=True)) / std
        combined_queries = torch.clamp(combined_queries, -5, 5)

        return {
            "appearance_queries": appearance_queries,
            "positional_queries": positional_queries,
            "random_queries": random_queries,
            "combined_queries": combined_queries,
            "boxes": boxes,
            "phrases": phrases,
            "masks": masks,
            "polygons": polygons
        }

    # Optionally, you could comment out or remove the generate_batch() method
    # if you want to enforce strictly sequential processing.
    # def generate_batch(self, pil_images, text_prompt):
    #     results = []
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(self.generate, img, text_prompt) for img in pil_images]
    #         for future in concurrent.futures.as_completed(futures):
    #             results.append(future.result())
    #     return results

############################################
# Global Instance (Optional)
############################################
_global_query_generator = None

def get_query_generator(device='cuda' if torch.cuda.is_available() else 'cpu', num_queries=300, hidden_dim=256):
    """
    Returns a cached QueryGenerator instance to ensure heavy initialization happens only once.
    """
    global _global_query_generator
    if _global_query_generator is None:
        _global_query_generator = QueryGenerator(device=device, num_queries=num_queries, hidden_dim=hidden_dim)
    return _global_query_generator
