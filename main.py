#!/usr/bin/env python
"""
Conditional DETR training and evaluation script.
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from utils.sam.sam_wrapper import SAMWrapper
from utils.grounding_dino.grounding_dino_wrapper import GroundingDINOWrapper

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util.misc import nested_tensor_from_tensor_list

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure main logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args_parser():
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='Initial learning rate for the backbone')
    parser.add_argument('--batch_size', default=6, type=int, help='Batch size per GPU')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of total epochs to run')
    parser.add_argument('--lr_drop', default=40, type=int, help='Epoch number to drop the learning rate')
    parser.add_argument('--clip_max_norm', default=1.0, type=float, help='Gradient clipping max norm')
    parser.add_argument('--num_classes', default=91, type=int, help="Number of object classes including background")
    parser.add_argument('--max_grounding_queries', default=50, type=int,
                        help='Maximum number of Grounding DINO queries per image.')
    parser.add_argument('--max_polygon_queries', default=50, type=int,
                        help='Maximum number of SAM polygon queries per image.')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="Replace stride with dilation in the last conv block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Positional embedding for the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoder layers in the Transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoder layers in the Transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Size of the feedforward layers in Transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Embedding dimension of the Transformer")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout in the Transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the Transformer attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--grounding_dino_config', type=str,
                        help='Path to Grounding DINO config file')
    parser.add_argument('--grounding_dino_checkpoint', type=str,
                        help='Path to Grounding DINO checkpoint file')
    parser.add_argument('--sam_checkpoint', type=str,
                        help='Path to SAM checkpoint file')
    parser.add_argument('--conditional_detr_checkpoint', type=str,
                        help='Path to the Conditional DETR checkpoint file')
    parser.add_argument('--use_grounding_dino', action='store_true',
                        help='Use Grounding DINO for dynamic queries')
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM if required')
    parser.add_argument('--return_intermediate_dec', action='store_true',
                        help='Return intermediate decoder layers')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if provided")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disable auxiliary decoding losses")
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help="Label smoothing coefficient for classification loss")
    parser.add_argument('--set_cost_class', default=0.5, type=float,
                        help="Class coefficient in matching cost")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float,
                        help="L1 box coefficient in matching cost")
    parser.add_argument('--set_cost_giou', default=2.0, type=float,
                        help="GIoU coefficient in matching cost")
    parser.add_argument("--iou_threshold", type=float, default=0.35, help="IoU threshold for matcher")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=5.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=1.0, type=float)
    parser.add_argument('--giou_loss_coef', default=1.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--amp', action='store_true', help='Use Automatic Mixed Precision (AMP) if True')
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, help='Path to coco dataset')
    parser.add_argument('--coco_panoptic_path', type=str, help='Path to coco panoptic dataset')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='', help='Path for saving outputs (empty for none)')
    parser.add_argument('--device', default='cuda', help='Training/testing device')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training setup')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode for detailed logging and visualization')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help="IoU threshold for matcher")
    parser.add_argument('--num_random_queries', default=50, type=int,
                        help="Number of learnable random queries.")
    return parser


def visualize_predictions(model, postprocessor, dataset, device, num_images=5, score_threshold=0.3, save_dir=None):
    model.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for idx in range(num_images):
        img, _ = dataset[idx]
        samples = nested_tensor_from_tensor_list([img]).to(device)
        outputs = model(samples)
        logger.debug(f"Raw logits (first 5): {outputs['pred_logits'][0][:5]}")
        logger.debug(f"Raw boxes (first 5): {outputs['pred_boxes'][0][:5]}")
        image_size = torch.as_tensor(img.shape[-2:], dtype=torch.float32, device=device).unsqueeze(0)
        results = postprocessor["bbox"](outputs, image_size)
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        if len(results) == 0:
            ax.text(10, 20, "No predictions", color="red", fontsize=14, backgroundcolor="white")
        else:
            pred = results[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f"{label}:{score:.2f}", color="yellow", fontsize=10)
        plt.title(f"Image {idx} Predictions")
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"predictions_{idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close(fig)


def main(args):
    utils.init_distributed_mode(args)
    logger.info(f"Git commit: {utils.get_sha()}")
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    logger.info(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize SAM wrapper if enabled.
    sam_wrapper = None
    if args.use_sam:
        if not args.sam_checkpoint:
            raise ValueError("SAM is enabled but 'sam_checkpoint' is not provided.")
        sam_wrapper = SAMWrapper(sam_checkpoint=args.sam_checkpoint, model_type="vit_l", device=device, debug=args.debug)
        sam_wrapper.to(device)

    # Initialize Grounding DINO wrapper if enabled.
    grounding_dino_wrapper = None
    if args.use_grounding_dino:
        if not args.grounding_dino_checkpoint or not args.grounding_dino_config:
            raise ValueError("Missing GroundingDINO config/checkpoint.")
        grounding_dino_wrapper = GroundingDINOWrapper(
            config_path=args.grounding_dino_config,
            checkpoint_path=args.grounding_dino_checkpoint,
            query_dim=args.hidden_dim,
            box_threshold=0.05,
            device=device,
            debug=args.debug,
            sam_wrapper=sam_wrapper
        )
        grounding_dino_wrapper.to(device)
        args.grounding_dino_wrapper = grounding_dino_wrapper

    # Build the model, criterion, and postprocessors.
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    criterion.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # Use a cosine annealing scheduler with T_max = total epochs and a small eta_min.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Build datasets.
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    sampler_val = (DistributedSampler(dataset_val, shuffle=False)
                   if args.distributed else torch.utils.data.SequentialSampler(dataset_val))
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    base_ds = get_coco_api_from_dataset(dataset_val)

    # Optionally resume from checkpoint.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and all(k in checkpoint for k in ['optimizer', 'lr_scheduler', 'epoch']):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.debug:
                logger.debug(f"Resumed from checkpoint {args.resume} at epoch {args.start_epoch}")
    # ----------------------------------------------
    # Debug: Run a single sample through the model.
    model.eval()
    sample_img, _ = dataset_val[0]  # use first validation sample
    samples = nested_tensor_from_tensor_list([sample_img]).to(device)
    with torch.no_grad():
        debug_outputs = model(samples)
    logger.info("\n[DEBUG] Single-sample forward pass:")
    logger.info("   pred_logits shape: {}".format(debug_outputs["pred_logits"].shape))
    logger.info("   pred_boxes shape: {}".format(debug_outputs["pred_boxes"].shape))
    softmax_preds = debug_outputs["pred_logits"].softmax(-1)
    logger.info("   Predicted class probabilities (first 10 queries):")
    logger.info(softmax_preds[0, :10, :])
    logger.info("   Raw bounding box coords (first 10 queries):")
    logger.info(debug_outputs["pred_boxes"][0, :10, :])
    # ----------------------------------------------

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval,
                                 Path(args.output_dir) / "eval.pth")
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)
        # In our training loop, before calling the model we extract the original images
        # as NumPy arrays (H, W, 3) so that the GroundingDINOWrapper can use them.
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device,
                                      epoch, max_norm=args.clip_max_norm, args=args)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val,
                                              base_ds, device, args=args)
        # We step the LR scheduler using the test loss.
        if isinstance(test_stats, dict) and "loss" in test_stats and isinstance(test_stats["loss"], (int, float)):
            lr_scheduler.step(test_stats["loss"])
        else:
            logger.warning("'loss' key missing or invalid in test_stats. Skipping LR scheduler step.")
        if args.output_dir:
            checkpoint_paths = [Path(args.output_dir) / "checkpoint.pth"]
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(Path(args.output_dir) / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                }, checkpoint_path)
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **{f"test_{k}": v for k, v in test_stats.items()},
                     "epoch": epoch,
                     "n_parameters": n_parameters}
        if args.output_dir and utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: " + total_time_str)
    visualize_save_dir = os.path.join(args.output_dir, "visual") if args.output_dir else None
    logger.info(f"Running visualization and saving images to {visualize_save_dir}")
    visualize_predictions(model, postprocessors, dataset_val, device, num_images=3,
                          score_threshold=0.1, save_dir=visualize_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Conditional DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
