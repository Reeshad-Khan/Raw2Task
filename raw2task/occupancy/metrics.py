"""Voxel occupancy metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def update_voxel_confusion(
    conf: np.ndarray,
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> None:
    pred_np = pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
    target_np = target.detach().cpu().numpy().astype(np.int64).reshape(-1)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    valid_np &= target_np != int(ignore_index)
    valid_np &= target_np >= 0
    valid_np &= target_np < int(num_classes)
    valid_np &= pred_np >= 0
    valid_np &= pred_np < int(num_classes)
    if not valid_np.any():
        return
    idx = target_np[valid_np] * int(num_classes) + pred_np[valid_np]
    conf += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def summarize_voxel_confusion(conf: np.ndarray) -> Dict[str, Any]:
    tp = np.diag(conf).astype(np.float64)
    pos_gt = conf.sum(axis=1).astype(np.float64)
    pos_pred = conf.sum(axis=0).astype(np.float64)
    union = pos_gt + pos_pred - tp
    iou = tp / np.maximum(union, 1.0)
    present = union > 0
    miou = float(iou[present].mean()) if present.any() else 0.0
    acc = float(tp.sum() / max(conf.sum(), 1))

    occupied_ids = [i for i in range(1, conf.shape[0])]
    if occupied_ids:
        occ_tp = conf[np.ix_(occupied_ids, occupied_ids)].sum()
        occ_gt = conf[occupied_ids, :].sum()
        occ_pred = conf[:, occupied_ids].sum()
        occ_iou = float(occ_tp / max(occ_gt + occ_pred - occ_tp, 1))
    else:
        occ_iou = 0.0

    return {
        "voxel_acc": acc,
        "mIoU": miou,
        "occupied_IoU": occ_iou,
        "per_class_IoU": [float(x) for x in iou],
        "per_class_support": [int(x) for x in pos_gt.tolist()],
        "per_class_pred": [int(x) for x in pos_pred.tolist()],
        "per_class_union": [int(x) for x in union.tolist()],
    }
