import math
import sys
import time
import datetime
import torch
import torch.nn.functional as F
from torch import nn
from typing import Iterable, Tuple, Dict
import numpy as np
import torchvision.ops as ops
import logging

import util.misc as utils
from util.misc import nested_tensor_from_tensor_list, MetricLogger
from datasets.coco_eval import CocoEvaluator  
from utils.grounding_dino.grounding_dino_wrapper import GroundingDINOWrapper
from utils.sam.sam_wrapper import SAMWrapper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 1.0,
    args=None
) -> Dict[str, float]:
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    # Initialize GroundingDINOWrapper once per epoch (if enabled)
    grounding_dino = None
    if getattr(args, "use_grounding_dino", False):
        grounding_dino = GroundingDINOWrapper(
            config_path=args.grounding_dino_config,
            checkpoint_path=args.grounding_dino_checkpoint,
            query_dim=args.hidden_dim,
            box_threshold=0.05,
            device=device,
            debug=True,
            sam_wrapper=args.sam_checkpoint and SAMWrapper(sam_checkpoint=args.sam_checkpoint, model_type="vit_l", device=device, debug=args.debug)
        )
        grounding_dino.to(device)
    
    for step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move images and targets to device
        images = [img.to(device) for img in images]
        adjusted_targets = []
        for t in targets:
            adjusted_t = {}
            for k, v in t.items():
                if k == "labels":
                    adjusted_t[k] = torch.clamp(v.clone().to(device), min=0, max=args.num_classes - 1)
                else:
                    adjusted_t[k] = v.to(device)
            adjusted_targets.append(adjusted_t)
        targets = adjusted_targets

        # Convert images to a nested tensor and also extract original images as NumPy arrays
        samples = nested_tensor_from_tensor_list(images).to(device)
        original_images = [img.permute(1, 2, 0).cpu().numpy() for img in images]

        outputs = model(samples)

        # Log positive matches (using your matcher)
        indices = criterion.matcher(outputs, targets)
        positive_matches = sum([len(src_idx) for (src_idx, _) in indices])
        logger.info(f"[DEBUG] Step {step} - Number of positive matches in batch: {positive_matches}")

        # If GroundingDINOWrapper is enabled, call get_queries with the original images.
        if grounding_dino is not None:
            boxes, dino_queries, sam_masks = grounding_dino.get_queries(samples.tensors, "object detection", original_images)
            if isinstance(dino_queries, list):
                for i, dq in enumerate(dino_queries):
                    logger.info(f"[DEBUG] Image {i} - GroundingDINO query shape: {dq.shape}")
            else:
                logger.info(f"[DEBUG] GroundingDINO queries shape: {dino_queries.shape}")

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        for loss_name, loss_val in loss_dict.items():
            metric_logger.update(**{loss_name: loss_val.item()})
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training.")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    postprocessors: dict,
    data_loader: Iterable,
    base_ds,
    device: torch.device,
    args=None
) -> Tuple[Dict[str, float], CocoEvaluator]:
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    coco_evaluator = CocoEvaluator(base_ds, ["bbox"])

    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 10, "Validation:")):
        images = [img.to(device) for img in images]
        adjusted_targets = []
        for t in targets:
            adjusted_t = {}
            for k, v in t.items():
                if k == "labels":
                    adjusted_labels = v.clone().to(device)
                    adjusted_labels = torch.clamp(adjusted_labels, min=0, max=args.num_classes - 1)
                    adjusted_t[k] = adjusted_labels
                else:
                    adjusted_t[k] = v.to(device)
            adjusted_targets.append(adjusted_t)
        targets = adjusted_targets

        samples = nested_tensor_from_tensor_list(images).to(device)
        outputs = model(samples)
        logger.debug(f"Model output keys: {list(outputs.keys())}")

        loss_dict = criterion(outputs, targets)
        image_sizes = [img.shape[-2:] for img in images]
        target_sizes = torch.as_tensor(image_sizes, dtype=torch.float32, device=device)
        results = postprocessors["bbox"](outputs, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        logger.debug(f"Converted evaluation results: {res}")

        coco_evaluator.update(res)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        metric_logger.update(loss=losses.item())

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    val_loss = metric_logger.meters["loss"].global_avg if "loss" in metric_logger.meters else None

    stats = coco_evaluator.coco_eval["bbox"].stats
    test_stats = {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "loss": val_loss,
    }
    return test_stats, coco_evaluator
