# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """Computes assignment between network predictions and targets.

    - Targets do not include "no-object".
    - More predictions than targets → some remain unmatched.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Initializes matcher with cost weights.

        Args:
            cost_class (float): Weight for classification cost.
            cost_bbox (float): Weight for L1 distance of box coordinates.
            cost_giou (float): Weight for GIoU loss of boxes.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs cannot be zero."

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the Hungarian matching.

        Args:
            outputs (dict): Contains:
                - "pred_logits": [batch_size, num_queries, num_classes] (classification logits)
                - "pred_boxes": [batch_size, num_queries, 4] (predicted bounding boxes)
            targets (list): List of dictionaries, one per batch:
                - "labels": [num_target_boxes] (ground-truth class labels)
                - "boxes": [num_target_boxes, 4] (ground-truth bounding boxes)

        Returns:
            List of size batch_size containing (index_i, index_j):
                - `index_i`: indices of selected predictions
                - `index_j`: indices of corresponding targets
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Debugging: Verify model output shape
        #print(f"DEBUG: outputs['pred_logits'].shape: {outputs['pred_logits'].shape}")

        # Flatten outputs to compute cost matrices in batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Handle empty targets case
        if all(len(v["labels"]) == 0 for v in targets):
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        # Concatenate target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Debugging: Print `tgt_ids` unique values
       # print(f"DEBUG: tgt_ids unique values: {torch.unique(tgt_ids)}")
        #print(f"DEBUG: max tgt_ids: {tgt_ids.max().item()}, expected max: {out_prob.shape[1]-1}")

        # Ensure labels are within range
        if tgt_ids.max().item() >= out_prob.shape[1]:
            #print(f"WARNING: Clamping tgt_ids to max {out_prob.shape[1] - 1}")
            tgt_ids = torch.clamp(tgt_ids, max=out_prob.shape[1] - 1)

        # Compute the classification cost using focal loss
        alpha = 0.25
        gamma =2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute L1 cost between predicted and target boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Get number of targets in each batch
        sizes = [len(v["boxes"]) for v in targets]

        # Perform Hungarian matching for each batch element
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    """Builds the HungarianMatcher based on provided arguments."""
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
