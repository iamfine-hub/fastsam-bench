#!/usr/bin/env python3
"""
FastSAM Everything-Mode Evaluation on ADE20K validation.
Preprocessing: long side→512, top-left pad to 512×512 (SAM3 protocol)
conf=0.4, iou=0.9 (NMS), masks truncated to top 300.

Metrics:
  - Instance: F-Score, Instance mIoU, Hungarian matching (same as COCO)
  - Semantic: Foreground IoU, Pixel Precision / Recall
  - Efficiency: Params, FLOPs, Latency
"""

import argparse
import base64
import json
import logging
import os
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from fastsam import FastSAM

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ======================== 配置 ========================
CONFIG = {
    "model_path": "FastSAM-x.pt",
    "img_dir": "/root/autodl-tmp/datasets/ADE20K/ADEChallengeData2016/images/validation",
    "ann_dir": "/root/autodl-tmp/datasets/ADE20K/ADEChallengeData2016/annotations/validation",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "conf": 0.4,
    "iou": 0.9,
    "imgsz": 1024,
    "max_masks": 300,
    "output_json": "fastsam_ade_preds.json",
    "match_iou_thresh": 0.5,
    "min_area": 500,
    "bench_warmup": 5,
    "bench_repeat": 20,
}
# ======================================================


def preprocess_image(image_bgr):
    """
    SAM3 protocol:
      1) long side → 512, keep aspect ratio
      2) top-left pad to 512×512 (black)
      3) stretch 512×512 → 1024×1024
    Returns:
      img_rgb:       np.ndarray (1024, 1024, 3) uint8 RGB
      scale_1024:    scale factor from original coords → 1024 space
      new_h_1024:    content height in 1024 space
      new_w_1024:    content width in 1024 space
    """
    h, w = image_bgr.shape[:2]
    scale_512 = 512.0 / max(h, w)
    new_h, new_w = int(round(h * scale_512)), int(round(w * scale_512))

    img_resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_b = 512 - new_h
    pad_r = 512 - new_w
    img_512 = cv2.copyMakeBorder(
        img_resized, 0, pad_b, 0, pad_r,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )

    img_1024 = cv2.resize(img_512, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    new_h_1024 = int(round(new_h * 2.0))
    new_w_1024 = int(round(new_w * 2.0))
    scale_1024 = scale_512 * 2.0

    img_rgb = cv2.cvtColor(img_1024, cv2.COLOR_BGR2RGB)
    return img_rgb, scale_1024, new_h_1024, new_w_1024


def mask_to_orig(mask_1024, new_h_1024, new_w_1024, orig_h, orig_w):
    """Bilinear-resize content region from 1024 space → original image space."""
    mask_crop = mask_1024[:new_h_1024, :new_w_1024]
    mask_tensor = torch.as_tensor(mask_crop, dtype=torch.float32)[None, None, :, :]
    mask_resized = torch.nn.functional.interpolate(
        mask_tensor, (orig_h, orig_w), mode="bilinear", align_corners=False,
    )[0, 0]
    return mask_resized.numpy() >= 0.5


def extract_gt_instances(gt_mask):
    """
    Extract connected components from a binary GT mask as pseudo-instances.
    Uses cv2.connectedComponentsWithStats.
    gt_mask: (H, W) binary uint8, 0=bg, >0=fg
    Returns: list of (H, W) binary masks, one per component with area ≥ 1
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    instances = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1:
            continue
        instances.append((labels == i).astype(np.uint8))
    return instances


def match_one_image(pred_masks, gt_masks, iou_thresh):
    """
    Hungarian bipartite matching for one image.
    Cost = 1 - IoU; pairs with IoU < threshold are discarded after matching.
    Returns: (tp, n_pred, n_gt, matched_ious_list)
    """
    P = len(pred_masks)
    G = len(gt_masks)
    if P == 0 or G == 0:
        return 0, P, G, []

    iou_mat = np.zeros((P, G), dtype=np.float32)
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou_mat[i, j] = inter / union if union > 0 else 0.0

    cost_mat = 1.0 - iou_mat
    pred_idx, gt_idx = linear_sum_assignment(cost_mat)

    tp = 0
    ious = []
    for pi, gj in zip(pred_idx, gt_idx):
        iou_val = float(iou_mat[pi, gj])
        if iou_val >= iou_thresh:
            tp += 1
            ious.append(iou_val)

    return tp, P, G, ious


def benchmark_model(model, device, warmup, repeat):
    """Params, FLOPs, Latency benchmark.  Decomposes into preprocess / inference / postprocess."""
    print(f"\n{'='*50}")
    print("Model Benchmark")
    print(f"{'='*50}")

    # --- Params ---
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Params:     {total_params/1e6:.2f} M total, {trainable/1e6:.2f} M trainable")

    # --- FLOPs ---
    try:
        from thop import profile
        model.model = model.model.to(device)
        dummy = torch.randn(1, 3, 1024, 1024).to(device)
        macs, _ = profile(model.model, inputs=(dummy,), verbose=False)
        print(f"  FLOPs:      {macs*2/1e9:.2f} G")
    except ImportError:
        print("  FLOPs:      N/A (pip install thop)")
    except Exception as e:
        print(f"  FLOPs:      N/A ({e})")

    # --- Latency: decompose pipeline ---
    from fastsam.predict import FastSAMPredictor
    overrides = {"conf": 0.4, "iou": 0.9, "imgsz": 1024, "max_det": 300,
                 "retina_masks": True, "device": device, "mode": "predict"}
    predictor = FastSAMPredictor(overrides=overrides)
    predictor.setup_model(model=model.model)
    predictor.imgsz = (1024, 1024)

    dummy_pil = Image.fromarray(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    )
    dummy_np = np.array(dummy_pil)

    # --- warmup (full pipeline) ---
    devnull = open(os.devnull, "w")
    for _ in range(warmup):
        with redirect_stdout(devnull):
            im_b = predictor.preprocess([dummy_np])
            p = predictor.model(im_b)
            _ = predictor.postprocess(p, im_b, [dummy_np])
    if device == "cuda":
        torch.cuda.synchronize()

    # --- timed runs ---
    pre_list, inf_list, post_list = [], [], []
    for _ in range(repeat):
        # preprocess
        t0 = time.perf_counter()
        im_b = predictor.preprocess([dummy_np])
        if device == "cuda":
            torch.cuda.synchronize()
        pre_list.append((time.perf_counter() - t0) * 1000)

        # inference (GPU – torch.cuda.Event)
        if device == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            p = predictor.model(im_b)
            e.record()
            torch.cuda.synchronize()
            inf_list.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            p = predictor.model(im_b)
            inf_list.append((time.perf_counter() - t0) * 1000)

        # postprocess
        t0 = time.perf_counter()
        _ = predictor.postprocess(p, im_b, [dummy_np])
        post_list.append((time.perf_counter() - t0) * 1000)

    devnull.close()

    pre_avg = np.mean(pre_list)
    inf_avg = np.mean(inf_list)
    post_avg = np.mean(post_list)
    full_avg = pre_avg + inf_avg + post_avg

    print(f"  [Latency] preprocess       = {pre_avg:.2f} ms")
    print(f"  [Latency] inference (GPU)  = {inf_avg:.2f} ms  (torch.cuda.Event, n={repeat})")
    print(f"  [Latency] postprocess      = {post_avg:.2f} ms  (NMS + mask decode, CPU)")
    print(f"  [Latency] full pipeline    = {full_avg:.2f} ms  (sum of above)")
    print(f"{'='*50}")


def main(max_images=0):
    cfg = CONFIG

    # ---- load model + benchmark ----
    print(f"Loading FastSAM: {cfg['model_path']}  (device={cfg['device']})")
    model = FastSAM(cfg["model_path"])
    benchmark_model(model, cfg["device"], cfg["bench_warmup"], cfg["bench_repeat"])

    # ---- scan ADE20K validation images ----
    img_dir = Path(cfg["img_dir"])
    ann_dir = Path(cfg["ann_dir"])
    img_paths = sorted(img_dir.glob("*.jpg"))
    if max_images > 0:
        img_paths = img_paths[:max_images]
    print(f"\nLoading ADE20K: {cfg['img_dir']}")
    print(f"  Images: {len(img_paths)}, conf={cfg['conf']}, nms_iou={cfg['iou']}, max_masks={cfg['max_masks']}")
    print(f"  Annotations: {cfg['ann_dir']}\n")

    # ---- incremental accumulators (unfiltered) ----
    all_preds = []
    total_tp = 0
    total_pred = 0
    total_gt = 0
    all_matched_ious = []
    total_mask_count = 0
    valid_img_count = 0
    fail_count = 0
    no_det_count = 0
    total_fg_iou_sum = 0.0
    total_fg_prec_sum = 0.0
    total_fg_recall_sum = 0.0
    per_img_recall = []  # for AR@300

    # ---- incremental accumulators (area ≥ min_area) ----
    total_tp_f = 0
    total_pred_f = 0
    total_gt_f = 0
    all_matched_ious_f = []
    total_mask_count_f = 0
    total_fg_iou_sum_f = 0.0
    total_fg_prec_sum_f = 0.0
    total_fg_recall_sum_f = 0.0
    valid_img_count_f = 0
    per_img_recall_f = []  # for AR@300

    # ---- eval loop ----
    for idx, img_path in enumerate(tqdm(img_paths)):
        # Find matching annotation
        ann_path = ann_dir / f"{img_path.stem}.png"
        if not ann_path.exists():
            fail_count += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            fail_count += 1
            continue

        # Use PIL for ADE20K annotation (indexed PNG, OpenCV would convert to BGR)
        try:
            ann_idx = np.array(Image.open(str(ann_path)))
        except Exception:
            fail_count += 1
            continue

        orig_h, orig_w = bgr.shape[:2]

        # Resize annotation to match image if needed
        if ann_idx.shape[:2] != (orig_h, orig_w):
            ann_idx = cv2.resize(ann_idx, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        img_rgb, scale_1024, new_h_1024, new_w_1024 = preprocess_image(bgr)
        pil_img = Image.fromarray(img_rgb)

        try:
            results = model(
                pil_img,
                device=cfg["device"],
                retina_masks=True,
                imgsz=cfg["imgsz"],
                conf=cfg["conf"],
                iou=cfg["iou"],
            )
        except Exception:
            print(f"\n  [ERROR] FastSAM inference failed for {img_path}")
            traceback.print_exc()
            fail_count += 1
            continue

        # ---- GT instances: connected components from binary foreground ----
        gt_binary = (ann_idx > 0).astype(np.uint8)
        gt_instances = extract_gt_instances(gt_binary)

        if results is None or len(results) == 0 or results[0].masks is None:
            no_det_count += 1
            tp, n_p, n_g, _ = match_one_image([], gt_instances, cfg["match_iou_thresh"])
            total_tp += tp
            total_pred += n_p
            total_gt += n_g
            valid_img_count += 1
            total_fg_recall_sum += 0.0
            total_fg_prec_sum += 1.0 if len(gt_instances) == 0 else 0.0
            total_fg_iou_sum += 0.0
            per_img_recall.append(1.0 if n_g == 0 else 0.0)
            # filtered — same (no masks to filter)
            total_tp_f += tp
            total_pred_f += n_p
            total_gt_f += n_g
            valid_img_count_f += 1
            total_fg_recall_sum_f += 0.0
            total_fg_prec_sum_f += 1.0 if len(gt_instances) == 0 else 0.0
            total_fg_iou_sum_f += 0.0
            total_mask_count_f += 0
            per_img_recall_f.append(1.0 if n_g == 0 else 0.0)
            continue

        result = results[0]
        masks_tensor = result.masks.data
        scores_tensor = result.boxes.conf
        N = masks_tensor.shape[0]
        total_mask_count += N
        valid_img_count += 1

        order = torch.argsort(scores_tensor, descending=True)
        keep = order[: cfg["max_masks"]] if N > cfg["max_masks"] else order

        pred_masks = []
        pred_scores = []
        for i_idx in keep:
            mask_1024 = (masks_tensor[i_idx] > 0).cpu().numpy()
            mask_orig = mask_to_orig(mask_1024, new_h_1024, new_w_1024, orig_h, orig_w)
            score = float(scores_tensor[i_idx].cpu())
            pred_masks.append(mask_orig)
            pred_scores.append(score)

        # ---- split preds by area filter ----
        min_area = cfg["min_area"]

        def _compute_pixel_metrics(masks, gt_fg):
            if len(masks) > 0:
                pred_union = np.zeros((orig_h, orig_w), dtype=bool)
                for pm in masks:
                    pred_union = np.logical_or(pred_union, pm)
            else:
                pred_union = np.zeros((orig_h, orig_w), dtype=bool)
            inter = np.logical_and(pred_union, gt_fg).sum()
            union = np.logical_or(pred_union, gt_fg).sum()
            iou = inter / union if union > 0 else 1.0
            prec = inter / pred_union.sum() if pred_union.sum() > 0 else (1.0 if gt_fg.sum() == 0 else 0.0)
            rec = inter / gt_fg.sum() if gt_fg.sum() > 0 else 1.0
            return iou, prec, rec

        gt_fg = gt_binary.astype(bool)

        # --- unfiltered ---
        iou_u, prec_u, rec_u = _compute_pixel_metrics(pred_masks, gt_fg)
        total_fg_iou_sum += iou_u
        total_fg_prec_sum += prec_u
        total_fg_recall_sum += rec_u

        tp, n_p, n_g, matched_ious = match_one_image(
            pred_masks, gt_instances, cfg["match_iou_thresh"],
        )
        total_tp += tp
        total_pred += n_p
        total_gt += n_g
        all_matched_ious.extend(matched_ious)
        per_img_recall.append(tp / n_g if n_g > 0 else 1.0)

        # --- area-filtered ---
        pred_masks_f = [m for m in pred_masks if m.sum() >= min_area]
        total_mask_count_f += len(pred_masks_f)
        valid_img_count_f += 1

        iou_f, prec_f, rec_f = _compute_pixel_metrics(pred_masks_f, gt_fg)
        total_fg_iou_sum_f += iou_f
        total_fg_prec_sum_f += prec_f
        total_fg_recall_sum_f += rec_f

        tp_f, n_p_f, n_g_f, matched_ious_f = match_one_image(
            pred_masks_f, gt_instances, cfg["match_iou_thresh"],
        )
        total_tp_f += tp_f
        total_pred_f += n_p_f
        total_gt_f += n_g_f
        all_matched_ious_f.extend(matched_ious_f)
        per_img_recall_f.append(tp_f / n_g_f if n_g_f > 0 else 1.0)

        # ---- save preds (per image) ----
        for i_idx, (mask, score) in enumerate(zip(pred_masks, pred_scores)):
            all_preds.append({
                "image_idx": idx,
                "image_name": img_path.name,
                "mask_idx": i_idx,
                "score": score,
                "orig_h": orig_h,
                "orig_w": orig_w,
                "mask_area": int(mask.sum()),
            })

    # ---- compute instance metrics (unfiltered) ----
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f_score = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    instance_miou = np.mean(all_matched_ious) if all_matched_ious else 0.0
    avg_mask_count = total_mask_count / valid_img_count if valid_img_count > 0 else 0.0

    # ---- compute instance metrics (area ≥ min_area) ----
    precision_f = total_tp_f / total_pred_f if total_pred_f > 0 else 0.0
    recall_f = total_tp_f / total_gt_f if total_gt_f > 0 else 0.0
    f_score_f = (
        2.0 * precision_f * recall_f / (precision_f + recall_f)
        if (precision_f + recall_f) > 0 else 0.0
    )
    instance_miou_f = np.mean(all_matched_ious_f) if all_matched_ious_f else 0.0
    avg_mask_count_f = total_mask_count_f / valid_img_count_f if valid_img_count_f > 0 else 0.0

    # ---- AR@300 (per-image recall averaged) ----
    ar300 = np.mean(per_img_recall) if per_img_recall else 0.0
    ar300_f = np.mean(per_img_recall_f) if per_img_recall_f else 0.0

    # ---- compute pixel-level averages (unfiltered) ----
    num_evaluated = valid_img_count
    avg_fg_iou = total_fg_iou_sum / num_evaluated if num_evaluated > 0 else 0.0
    avg_fg_prec = total_fg_prec_sum / num_evaluated if num_evaluated > 0 else 0.0
    avg_fg_recall = total_fg_recall_sum / num_evaluated if num_evaluated > 0 else 0.0

    # ---- compute pixel-level averages (area ≥ min_area) ----
    num_evaluated_f = valid_img_count_f
    avg_fg_iou_f = total_fg_iou_sum_f / num_evaluated_f if num_evaluated_f > 0 else 0.0
    avg_fg_prec_f = total_fg_prec_sum_f / num_evaluated_f if num_evaluated_f > 0 else 0.0
    avg_fg_recall_f = total_fg_recall_sum_f / num_evaluated_f if num_evaluated_f > 0 else 0.0

    # ---- save preds ----
    with open(cfg["output_json"], "w") as f:
        json.dump(all_preds, f)

    # ---- final summary ----
    print(f"\n{'='*50}")
    print("ADE20K Everything-Mode Evaluation Results")
    print(f"{'='*50}")
    print(f"  Valid images:      {valid_img_count}  (fail: {fail_count}, no_det: {no_det_count})")
    print(f"  Avg masks/image (all):        {avg_mask_count:.2f}  (before truncation, max={cfg['max_masks']})")
    print(f"  Avg masks/image (≥{cfg['min_area']}px): {avg_mask_count_f:.2f}")
    print(f"\n  ---- Instance-Level Metrics (Hungarian, IoU≥{cfg['match_iou_thresh']}) ----")
    print(f"  {'':>20} {'All masks':>16} {'≥{0}px'.format(cfg['min_area']):>16}")
    print(f"  {'Precision:':>20} {precision:>16.4f} {precision_f:>16.4f}")
    print(f"  {'Recall:':>20} {recall:>16.4f} {recall_f:>16.4f}")
    print(f"  {'F-Score:':>20} {f_score:>16.4f} {f_score_f:>16.4f}")
    print(f"  {'Instance mIoU:':>20} {instance_miou:>16.4f} {instance_miou_f:>16.4f}")
    print(f"  {'AR@300:':>20} {ar300:>16.4f} {ar300_f:>16.4f}")
    print(f"  {'TP/Pred/GT:':>20} {total_tp}/{total_pred}/{total_gt:>16} {total_tp_f}/{total_pred_f}/{total_gt_f}")
    print(f"\n  ---- Pixel-Level Foreground Metrics ----")
    print(f"  {'':>20} {'All masks':>16} {'≥{0}px'.format(cfg['min_area']):>16}")
    print(f"  {'Foreground IoU:':>20} {avg_fg_iou:>16.4f} {avg_fg_iou_f:>16.4f}")
    print(f"  {'Pixel Precision:':>20} {avg_fg_prec:>16.4f} {avg_fg_prec_f:>16.4f}")
    print(f"  {'Pixel Recall:':>20} {avg_fg_recall:>16.4f} {avg_fg_recall_f:>16.4f}")
    print(f"\n  Preds saved:       {cfg['output_json']}  ({len(all_preds)} entries)")
    print(f"{'='*50}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit images for quick test, 0 = all")
    args = parser.parse_args()
    main(max_images=args.max_images)
