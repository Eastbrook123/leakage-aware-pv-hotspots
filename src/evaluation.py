"""Independent detection-evaluation utilities (no dependency on the training framework).

Matching rule: predictions are sorted by confidence (descending, across the whole set for AP,
per image for counting); each prediction is matched to the unmatched ground-truth box of the
same class with the highest IoU; it is a true positive if that IoU >= ``iou_thr``, otherwise a
false positive. Unmatched ground-truth boxes are false negatives.

AP uses all-point interpolation of the precision envelope (COCO-style 101-point sampling is
also provided). mAP@0.5:0.95 averages AP over IoU thresholds 0.50, 0.55, ..., 0.95.
"""
from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Greedy match for one image. Returns (tp_flags in score order, scores sorted, n_gt, matched_iou)."""
    order = np.argsort(-np.asarray(pred_scores, dtype=float))
    pb = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)[order]
    ps = np.asarray(pred_scores, dtype=float)[order]
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    used = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(pb), dtype=bool)
    ious = box_iou(pb, gt)
    matched = []
    for i in range(len(pb)):
        if len(gt) == 0:
            break
        cand = np.where(used, -1.0, ious[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            used[j] = True
            tp[i] = True
            matched.append(float(cand[j]))
    return tp, ps, len(gt), matched


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int, method: str = "continuous"):
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([])
    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    if method == "coco101":
        x = np.linspace(0, 1, 101)
        ap = float(np.mean(np.interp(x, mrec, mpre, right=0)))
    else:
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, precision, recall


def evaluate_dataset(preds: list, gts: list, conf_threshold: float = 0.25, iou_thr: float = 0.5) -> dict:
    """``preds``: list per image of (boxes (N,4), scores (N,)); ``gts``: list per image of boxes (M,4).

    Returns AP50, AP50:95, and P/R/F1/TP/FP/FN at ``conf_threshold`` (IoU ``iou_thr``), plus
    per-image counts and the precision/recall/F1-vs-confidence curve for threshold selection.
    """
    n_gt = sum(len(g) for g in gts)
    aps = {}
    for t in np.round(np.arange(0.5, 0.96, 0.05), 2):
        all_tp, all_s = [], []
        for (pb, ps), gb in zip(preds, gts):
            tp, s, _, _ = match_image(pb, ps, gb, iou_thr=float(t))
            all_tp.append(tp)
            all_s.append(s)
        tp = np.concatenate(all_tp) if all_tp else np.array([], dtype=bool)
        sc = np.concatenate(all_s) if all_s else np.array([])
        aps[float(t)] = average_precision(tp, sc, n_gt)[0]

    per_image = []
    TP = FP = FN = 0
    all_tp50, all_s50 = [], []
    for k, ((pb, ps), gb) in enumerate(zip(preds, gts)):
        tp_all, s_all, _, _ = match_image(pb, ps, gb, iou_thr=iou_thr)
        all_tp50.append(tp_all)
        all_s50.append(s_all)
        keep = np.asarray(ps) >= conf_threshold
        pbk = np.asarray(pb).reshape(-1, 4)[keep]
        psk = np.asarray(ps)[keep]
        tp, _, ng, _ = match_image(pbk, psk, gb, iou_thr=iou_thr)
        t, f = int(tp.sum()), int(len(tp) - tp.sum())
        fn = ng - t
        TP, FP, FN = TP + t, FP + f, FN + fn
        per_image.append({"index": k, "tp": t, "fp": f, "fn": fn, "n_gt": ng})
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0

    # threshold sweep (from the IoU-0.5 matching of all predictions)
    tp50 = np.concatenate(all_tp50) if all_tp50 else np.array([], dtype=bool)
    s50 = np.concatenate(all_s50) if all_s50 else np.array([])
    curve = []
    for c in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = s50 >= c
        t = int(tp50[m].sum()); f = int(m.sum() - t)
        p = t / (t + f) if t + f else 1.0
        r = t / n_gt if n_gt else 0.0
        curve.append({"conf": float(c), "precision": p, "recall": r,
                      "f1": 2 * p * r / (p + r) if p + r else 0.0})

    img_with_gt = [x for x in per_image if x["n_gt"] > 0]
    img_no_gt = [x for x in per_image if x["n_gt"] == 0]
    return {
        "iou_threshold": iou_thr, "conf_threshold": conf_threshold,
        "images": len(gts), "gt_boxes": n_gt,
        "tp": TP, "fp": FP, "fn": FN, "precision": P, "recall": R, "f1": F1,
        "ap50": aps.get(0.5), "ap50_95": float(np.nanmean(list(aps.values()))) if aps else None,
        "ap_by_iou": aps,
        "image_level": {
            "images_with_hotspot": len(img_with_gt),
            "hit_rate": (sum(1 for x in img_with_gt if x["tp"] > 0) / len(img_with_gt)) if img_with_gt else None,
            "images_without_hotspot": len(img_no_gt),
            "false_alarm_rate": (sum(1 for x in img_no_gt if x["fp"] > 0) / len(img_no_gt)) if img_no_gt else None,
        },
        "threshold_curve": curve,
        "per_image": per_image,
    }
