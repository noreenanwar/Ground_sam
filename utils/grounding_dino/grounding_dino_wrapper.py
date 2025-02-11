#!/usr/bin/env python
"""
Modified GroundingDINOWrapper for Conditional DETR

This wrapper loads the pretrained Grounding DINO model (and optionally a SAM wrapper),
performs image preprocessing, and generates query embeddings from both the model’s outputs 
and SAM (polygon embeddings). When a SAM wrapper is provided the original images (as NumPy RGB arrays)
are used to predict segmentation masks. Those masks are converted to polygon embeddings (projected 
into the same query space) and concatenated with the queries from DINO.

If you provide a generic text prompt (e.g. "object detection"), it will automatically be replaced 
with a detailed prompt listing the COCO categories.
"""

import torch
from torch import nn
import numpy as np
import cv2
import traceback
import logging
from typing import Optional, List, Tuple
from torchvision import transforms

# Import Grounding DINO components
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

# Import SAMWrapper from our module.
from utils.sam.sam_wrapper import SAMWrapper

# Set up logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class GroundingDINOWrapper(nn.Module):
    """
    Modified wrapper for Grounding DINO (and optionally SAM) that generates query embeddings.
    
    When a SAM wrapper is provided, the original images (as NumPy RGB arrays) are used to compute 
    segmentation masks. These masks are converted into polygon embeddings (projected into the query space)
    and concatenated with the queries produced by the DINO model.
    """
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        query_dim: int = 256,
        box_threshold: float = 0.25,
        device: str = "cuda",
        debug: bool = False,
        sam_wrapper: Optional[SAMWrapper] = None
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.debug = debug
        self.box_threshold = box_threshold
        self.query_dim = query_dim
        self.sam_wrapper = sam_wrapper

        # Load the Grounding DINO model.
        self.model, self.config = self._load_model(config_path, checkpoint_path)
        self.model.to(self.device).eval()

        # Normalization transform (using ImageNet statistics)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Linear layers for projecting logits and polygon coordinates into the query space.
        self.logits_to_query = nn.Linear(256, query_dim).to(self.device)
        nn.init.xavier_uniform_(self.logits_to_query.weight)
        nn.init.constant_(self.logits_to_query.bias, 0.0)

        self.polygon_to_query = nn.Linear(8, query_dim).to(self.device)
        nn.init.xavier_uniform_(self.polygon_to_query.weight)
        nn.init.constant_(self.polygon_to_query.bias, 0.0)

        if self.debug:
            logger.debug("GroundingDINOWrapper initialized successfully.")

    def _load_model(self, config_path: str, checkpoint_path: str):
        try:
            args = SLConfig.fromfile(config_path)
            args.device = str(self.device)
            model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            return model, args
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO model: {e}")
            traceback.print_exc()
            raise

    @torch.no_grad()
    def _process_batch(self, image_tensor: torch.Tensor, text_prompts: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        try:
            image_tensor = image_tensor.to(self.device, non_blocking=True)
            if image_tensor.max() > 1:
                image_tensor = image_tensor / 255.0

            for b in range(image_tensor.size(0)):
                image_tensor[b] = self.normalize(image_tensor[b])
            
            # Substitute generic prompt with detailed COCO categories prompt if needed.
            for i, prompt in enumerate(text_prompts):
                if prompt.strip().lower() == "object detection" or len(prompt.strip().split()) < 5:
                    if self.debug:
                        logger.info("Generic text prompt detected; substituting with detailed COCO categories prompt.")
                    text_prompts[i] = (
                        "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . "
                        "traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . "
                        "dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . "
                        "handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . "
                        "baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . "
                        "wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . "
                        "broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . "
                        "bed . dining table . toilet . TV . laptop . mouse . remote . keyboard . cell phone . "
                        "microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . "
                        "teddy bear . hair drier . toothbrush"
                    )
            
            outputs = self.model(image_tensor, captions=text_prompts)
            logits = outputs["pred_logits"].sigmoid()
            boxes = outputs["pred_boxes"]

            boxes_list = []
            logits_list = []
            B = image_tensor.size(0)
            for b_idx in range(B):
                log_b = logits[b_idx]
                box_b = boxes[b_idx]
                conf_b, _ = log_b.max(dim=1)
                if self.debug:
                    max_conf = conf_b.max().item() if conf_b.numel() > 0 else 0.0
                    logger.debug(f"Image {b_idx}: max confidence = {max_conf:.4f}")
                keep = conf_b > self.box_threshold
                if keep.sum() == 0:
                    if self.debug:
                        logger.warning(f"Image {b_idx}: No queries passed threshold {self.box_threshold}. Using all queries as fallback.")
                    keep = torch.ones_like(conf_b, dtype=torch.bool)
                filtered_boxes = box_b[keep].to(self.device)
                filtered_logits = log_b[keep].to(self.device)
                boxes_list.append(filtered_boxes)
                logits_list.append(filtered_logits)
            return boxes_list, logits_list

        except Exception as e:
            logger.error(f"Error in _process_batch: {e}")
            traceback.print_exc()
            raise

    @torch.no_grad()
    def convert_masks_to_polygon_embeddings(self, masks: List[np.ndarray]) -> torch.Tensor:
        embeddings = []
        for mask in masks:
            if mask is None:
                continue
            poly = self.sam_wrapper.convert_mask_to_polygon(mask)
            if poly is not None:
                embedding = self.polygon_to_query(torch.tensor(poly, dtype=torch.float32, device=self.device))
                embeddings.append(embedding)
        if embeddings:
            return torch.stack(embeddings, dim=0)
        else:
            if self.debug:
                logger.debug("No polygon embeddings generated; returning empty tensor.")
            return torch.empty((0, self.query_dim), device=self.device)
    @torch.no_grad()
    def _predict_sam_masks_for_single(self, boxes: torch.Tensor, original_image: np.ndarray) -> List[np.ndarray]:
        if self.sam_wrapper is None:
            return []
        h, w, _ = original_image.shape
        logger.debug(f"Original image shape for SAM: {original_image.shape}")
        boxes_list = []
        for box in boxes:
            # Box is assumed to be [cx, cy, bw, bh] in normalized coordinates.
            cx, cy, bw, bh = box.cpu().numpy()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            # --- NEW: Clamp coordinates to valid image dimensions ---
            x1 = np.clip(x1, 0, w)
            y1 = np.clip(y1, 0, h)
            x2 = np.clip(x2, 0, w)
            y2 = np.clip(y2, 0, h)
            # ----------------------------------------------------------
            logger.debug(f"Converted box: {[x1, y1, x2, y2]}")
            boxes_list.append([x1, y1, x2, y2])
        self.sam_wrapper.set_image(original_image)
        masks = []
        for box in boxes_list:
            box_np = np.array([box], dtype=np.float32)
            try:
                mask_list = self.sam_wrapper.predict_masks(box_np)
            except Exception as e:
                logger.error(f"Error predicting mask for box {box}: {e}")
                continue
            if mask_list is not None and mask_list.shape[0] > 0:
                masks.append(mask_list[0])
        return masks

    @torch.no_grad()
    def get_queries(self, image_tensor: torch.Tensor, text_prompt: str, original_images: Optional[List[np.ndarray]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[np.ndarray]]]:
        B = image_tensor.size(0)
        text_prompts = [text_prompt] * B
        boxes_list, logits_list = self._process_batch(image_tensor, text_prompts)
        final_boxes = []
        final_embeds = []
        final_masks = []
        for b_idx in range(B):
            b_boxes = boxes_list[b_idx]
            b_logits = logits_list[b_idx]
            box_queries = torch.relu(self.logits_to_query(b_logits))
            mask_list_b = []
            poly_embeds = torch.empty((0, self.query_dim), device=self.device)
            if original_images is not None and len(original_images) == B and self.sam_wrapper is not None:
                sam_masks = self._predict_sam_masks_for_single(b_boxes, original_images[b_idx])
                mask_list_b = sam_masks
                if len(sam_masks) > 0:
                    poly_embeds = self.convert_masks_to_polygon_embeddings(sam_masks)
            combined = torch.cat([box_queries, poly_embeds], dim=0) if poly_embeds.size(0) > 0 else box_queries
            final_boxes.append(b_boxes)
            final_embeds.append(combined)
            final_masks.append(mask_list_b)
        return final_boxes, final_embeds, final_masks


###############################################################################
# Main block for testing.
###############################################################################
if __name__ == "__main__":
    config_path = "config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "checkpoints/groundingdino_swint_ogc.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam_checkpoint = "checkpoints/sam_vit_l_0b3195.pth"
    sam_wrapper = SAMWrapper(sam_checkpoint=sam_checkpoint, model_type="vit_l", device=device, debug=True)
    sam_wrapper.to(device)
    
    wrapper = GroundingDINOWrapper(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        query_dim=256,
        box_threshold=0.05,
        device=device,
        debug=True,
        sam_wrapper=sam_wrapper
    )
    
    