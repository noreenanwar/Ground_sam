#!/usr/bin/env python
"""
Conditional DETR (modified)

This version has been modified so that:
  - The classification head outputs (num_classes + 1) logits (the last index is “no-object”).
  - The loss computation in SetCriterion.loss_labels uses all output channels.
  - The postprocessing correctly clamps the predicted boxes using tensor min/max.
  - The classification head bias is initialized to zero (to avoid an early bias toward "no-object").
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Dict, Optional, Tuple, Union

from util.misc import (
    NestedTensor, nested_tensor_from_tensor_list,
    accuracy, get_world_size, is_dist_avail_and_initialized
)
from util import box_ops

from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_transformer
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)

from utils.grounding_dino.grounding_dino_wrapper import GroundingDINOWrapper
from utils.sam.sam_wrapper import SAMWrapper

import numpy as np
import traceback
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
import cv2 


class ConditionalDETR(nn.Module):
    """
    Conditional DETR with three types of queries:
      1. Appearance-based queries from Grounding DINO.
      2. Positional queries (extracted via SAM inside GroundingDINOWrapper).
      3. Random (learnable) queries.
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, hidden_dim=256, 
                 grounding_dino_config=None, grounding_dino_checkpoint=None, sam_checkpoint=None,
                 max_grounding_queries: int = 100,
                 num_random_queries: int = 50,
                 debug: bool = False,
                 **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.max_grounding_queries = max_grounding_queries

        # Input projection: map backbone features to hidden_dim.
        self.input_proj = nn.Conv2d(getattr(backbone, 'num_channels', 2048), hidden_dim, kernel_size=1)

        # Instantiate the Grounding DINO wrapper (which also handles SAM if provided).
        self.grounding_dino = GroundingDINOWrapper(
            config_path=grounding_dino_config,
            checkpoint_path=grounding_dino_checkpoint,
            query_dim=hidden_dim,
            box_threshold=0.20,
            device="cuda",
            debug=debug,
            sam_wrapper=SAMWrapper(sam_checkpoint, device="cuda", debug=debug) if sam_checkpoint else None
        )

        # Learnable random queries.
        self.num_random_queries = num_random_queries
        self.random_queries = nn.Parameter(torch.zeros(self.num_random_queries, hidden_dim))
        nn.init.xavier_uniform_(self.random_queries)

        # Prediction heads.
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.aux_loss = aux_loss
        self._init_weights()

    def _init_weights(self):
        # Set classification head bias to zero to remove initial bias toward "no-object."
        nn.init.constant_(self.class_embed.bias, 0.0)
        nn.init.normal_(self.class_embed.weight, std=0.01)
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0)

    def forward(self, samples: Union[NestedTensor, torch.Tensor, List[torch.Tensor]]):
        # If samples is not already a NestedTensor, wrap it appropriately.
        if not hasattr(samples, 'tensors'):
            if isinstance(samples, torch.Tensor):
                if samples.dim() == 3:
                    # Single image [C, H, W]
                    samples = nested_tensor_from_tensor_list([samples])
                elif samples.dim() == 4:
                    # Batch of images [B, C, H, W]
                    B, C, H, W = samples.shape
                    mask = torch.zeros(B, H, W, dtype=torch.bool, device=samples.device)
                    samples = NestedTensor(samples, mask)
                else:
                    raise ValueError(f"Unsupported tensor dimension: {samples.dim()}. Expected 3 or 4.")
            else:
                # Assume it's a list of tensors (each [C, H, W])
                samples = nested_tensor_from_tensor_list(samples)
                
        features, pos = self.backbone(samples)
        src = features[-1].tensors   # [B, C, H, W]
        mask = features[-1].mask     # [B, H, W]

        src = self.input_proj(src)

        # Fixed text prompt for query generation.
        text_prompt = (
            "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . "
            "fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . "
            "elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . "
            "skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . "
            "tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . "
            "orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . "
            "dining table . toilet . TV . laptop . mouse . remote . keyboard . cell phone . microwave . oven . "
            "toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush"
        )

        boxes, dino_queries, _ = self.grounding_dino.get_queries(
            samples.tensors,
            text_prompt,
            original_images=[img for img in samples.original_images] if hasattr(samples, 'original_images') else None
        )
        bs = samples.tensors.size(0)
        padded_dino_queries = []
        for q in dino_queries:
            N_i = q.size(0)
            if N_i < self.max_grounding_queries:
                pad_size = self.max_grounding_queries - N_i
                padding = q.new_zeros(pad_size, q.size(1))
                q_padded = torch.cat([q, padding], dim=0)
            else:
                q_padded = q[:self.max_grounding_queries]
            padded_dino_queries.append(q_padded)
        # dino_queries_tensor shape: [max_grounding_queries, B, hidden_dim]
        dino_queries_tensor = torch.stack(padded_dino_queries, dim=0).transpose(0, 1)

        random_queries = self.random_queries.unsqueeze(1).expand(-1, bs, -1)

        combined_queries = torch.cat([dino_queries_tensor, random_queries], dim=0)

        hs, _ = self.transformer(src, mask, combined_queries, pos[-1])

        outputs_class = self.class_embed(hs)   # [num_layers, B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # normalized boxes
        return {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'aux_outputs': self._get_aux_outputs(outputs_class, outputs_coord) if self.aux_loss else None
        }

    def _get_aux_outputs(self, outputs_class, outputs_coord):
        return [{'pred_logits': cl, 'pred_boxes': bx}
                for cl, bx in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """
    A simple multi-layer perceptron.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim,
                      hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    """
    This class computes the loss for Conditional DETR.
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        super().__init__()
        self.num_classes = num_classes  
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
            dtype=src_logits.dtype, device=src_logits.device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                     alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks.device)
        target_masks = target_masks[tgt_idx]
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs and outputs['aux_outputs'] is not None:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue  # Skip intermediate mask loss for efficiency.
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Convert model output to COCO API format.
        """
        assert "pred_logits" in outputs and "pred_boxes" in outputs, "Missing required keys in outputs"
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = out_logits.softmax(-1)
        scores, labels = prob.max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_factor = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_factor[:, None, :]

        # Clamp boxes to image boundaries.
        min_val = torch.zeros_like(boxes)
        max_val = scale_factor[:, None, :].to(boxes.dtype)
        boxes = torch.clamp(boxes, min=min_val, max=max_val)

        results = []
        for s, l, b in zip(scores, labels, boxes):
            results.append({
                "scores": s,
                "labels": l,
                "boxes": b
            })
        return results


def build(args):
    """
    Build the Conditional DETR model, its criterion, and postprocessors.
    """
    if args.dataset_file == "coco":
        num_classes = 91
    elif args.dataset_file == "coco_panoptic":
        num_classes = 250
    else:
        num_classes = 20

    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    # Instantiate SAMWrapper if needed.
    sam_wrapper = SAMWrapper(
        sam_checkpoint=args.sam_checkpoint,
        model_type="vit_l",
        device=args.device,
        debug=args.debug
    ) if args.use_sam else None

    model = ConditionalDETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        debug=getattr(args, "debug", False),
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        grounding_dino_config=args.grounding_dino_config,
        hidden_dim=args.hidden_dim,
        max_grounding_queries=args.max_grounding_queries,
        num_random_queries=args.num_random_queries if hasattr(args, "num_random_queries") else 50,
        sam_checkpoint=args.sam_checkpoint
    )

    if getattr(args, "masks", False):
        from models.segmentation import DETRsegm, PostProcessSegm, PostProcessPanoptic
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef
    }
    if getattr(args, "masks", False):
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            for k, v in weight_dict.items():
                aux_weight_dict[f"{k}_{i}"] = v
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if getattr(args, "masks", False):
        losses.append("masks")

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=args.focal_alpha,
    )
    criterion.to(device)

    postprocessors = {"bbox": PostProcess()}
    if getattr(args, "masks", False):
        from models.segmentation import PostProcessSegm, PostProcessPanoptic
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
