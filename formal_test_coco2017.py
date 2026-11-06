#!/usr/bin/env python3
"""
FastSAM Everything-Mode Evaluation on COCO 2017 val.
Preprocessing: long side→512, top-left pad to 512×512 (SAM3 protocol)
conf=0.4, iou=0.9 (NMS), masks truncated to top 300.

Metrics: AR@300, avg mask count, F-Score, Instance mIoU, Params, FLOPs, Latency
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from contextlib import redirect_stdout
import numpy as np
import torch
import traceback
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from fastsam import FastSAM

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ======================== 配置 ========================
CONFIG = {
    "model_path": "FastSAM-x.pt",
    "img_dir": "/root/autodl-tmp/datasets/coco2017val/images/val2017",
    "ann_file": "/root/autodl-tmp/datasets/coco2017val/annotations/instances_val2017.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "conf": 0.4,
    "iou": 0.9,
    "imgsz": 1024,
    "max_masks": 300,
    "output_json": "fastsam_coco_preds.json",
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
    # crop content region (drop black padding)
    mask_crop = mask_1024[:new_h_1024, :new_w_1024]
    # bilinear resize to original resolution
    mask_tensor = torch.as_tensor(mask_crop, dtype=torch.float32)[None, None, :, :]
    mask_resized = torch.nn.functional.interpolate(
        mask_tensor, (orig_h, orig_w), mode="bilinear", align_corners=False,
    )[0, 0]
    return mask_resized.numpy() >= 0.5


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
    # Build a temporary predictor so we can time preprocess / inference / postprocess separately
    from fastsam.predict import FastSAMPredictor
    overrides = {"conf": 0.4, "iou": 0.9, "imgsz": 1024, "max_det": 300,
                 "retina_masks": True, "device": device, "mode": "predict"}
    predictor = FastSAMPredictor(overrides=overrides)
    predictor.setup_model(model=model.model)
    predictor.imgsz = (1024, 1024)  # set manually (normally done by setup_source)

    dummy_pil = Image.fromarray(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    )
    dummy_np = np.array(dummy_pil)  # (1024, 1024, 3) uint8 RGB

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

    # ---- load COCO ----
    print(f"Loading COCO: {cfg['ann_file']}")
    coco = COCO(cfg["ann_file"])
    img_ids = sorted(coco.getImgIds())
    if max_images > 0:
        img_ids = img_ids[:max_images]
    print(f"  Images: {len(img_ids)}, conf={cfg['conf']}, nms_iou={cfg['iou']}, max_masks={cfg['max_masks']}\n")

    # ---- incremental accumulators ----
    all_coco_preds = []
    total_tp, total_pred, total_gt = 0, 0, 0
    all_matched_ious = []
    total_mask_count = 0
    valid_img_count = 0
    fail_count = 0
    no_det_count = 0
    # pixel-level + AR@300 (unfiltered)
    total_fg_iou, total_fg_prec, total_fg_rec = 0.0, 0.0, 0.0
    per_img_recall = []
    # area-filtered
    total_tp_f, total_pred_f, total_gt_f = 0, 0, 0
    all_matched_ious_f = []
    total_mask_count_f = 0
    total_fg_iou_f, total_fg_prec_f, total_fg_rec_f = 0.0, 0.0, 0.0
    per_img_recall_f = []

    # ---- eval loop ----
    for img_id in tqdm(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(cfg["img_dir"], info["file_name"])
        orig_w, orig_h = info["width"], info["height"]

        bgr = cv2.imread(img_path)
        if bgr is None:
            fail_count += 1
            continue
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

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_masks = [(coco.annToMask(ann) > 0) for ann in anns]

        if results is None or len(results) == 0 or results[0].masks is None:
            no_det_count += 1
            n_g = len(gt_masks)
            total_gt += n_g
            total_gt_f += n_g
            per_img_recall.append(1.0 if n_g == 0 else 0.0)
            per_img_recall_f.append(1.0 if n_g == 0 else 0.0)
            total_fg_prec += 1.0 if n_g == 0 else 0.0
            total_fg_prec_f += 1.0 if n_g == 0 else 0.0
            valid_img_count += 1
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
        coco_preds_this = []
        for idx in keep:
            mask_1024 = (masks_tensor[idx] > 0).cpu().numpy()
            mask_orig = mask_to_orig(mask_1024, new_h_1024, new_w_1024, orig_h, orig_w)
            score = float(scores_tensor[idx].cpu())

            pred_masks.append(mask_orig)
            pred_scores.append(score)

            rle = mask_utils.encode(np.asfortranarray(mask_orig.astype(np.uint8)))
            if isinstance(rle["counts"], bytes):
                rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")
            coco_preds_this.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": score,
            })

        all_coco_preds.extend(coco_preds_this)

        # ---- GT foreground for pixel metrics ----
        gt_fg = np.zeros((orig_h, orig_w), dtype=bool)
        for gm in gt_masks:
            np.logical_or(gt_fg, gm, out=gt_fg)

        # ---- unfiltered pixel-level ----
        if pred_masks:
            pu = pred_masks[0].copy()
            for m in pred_masks[1:]:
                np.logical_or(pu, m, out=pu)
        else:
            pu = np.zeros((orig_h, orig_w), dtype=bool)
        inter = np.logical_and(pu, gt_fg).sum()
        ps, gs = pu.sum(), gt_fg.sum()
        denom = ps + gs - inter
        total_fg_iou += inter / denom if denom > 0 else 1.0
        total_fg_prec += inter / ps if ps > 0 else (1.0 if gs == 0 else 0.0)
        total_fg_rec += inter / gs if gs > 0 else 1.0

        # ---- unfiltered instance ----
        tp, n_p, n_g, matched_ious = match_one_image(
            pred_masks, gt_masks, cfg["match_iou_thresh"],
        )
        total_tp += tp
        total_pred += n_p
        total_gt += n_g
        all_matched_ious.extend(matched_ious)
        per_img_recall.append(tp / n_g if n_g > 0 else 1.0)

        # ---- area-filtered ----
        pm_f = [m for m in pred_masks if m.sum() >= cfg["min_area"]]
        total_mask_count_f += len(pm_f)

        if pm_f:
            pu_f = pm_f[0].copy()
            for m in pm_f[1:]:
                np.logical_or(pu_f, m, out=pu_f)
        else:
            pu_f = np.zeros((orig_h, orig_w), dtype=bool)
        inter_f = np.logical_and(pu_f, gt_fg).sum()
        ps_f, gs_f = pu_f.sum(), gt_fg.sum()
        denom_f = ps_f + gs_f - inter_f
        total_fg_iou_f += inter_f / denom_f if denom_f > 0 else 1.0
        total_fg_prec_f += inter_f / ps_f if ps_f > 0 else (1.0 if gs_f == 0 else 0.0)
        total_fg_rec_f += inter_f / gs_f if gs_f > 0 else 1.0

        tp_f, n_p_f, n_g_f, matched_ious_f = match_one_image(
            pm_f, gt_masks, cfg["match_iou_thresh"],
        )
        total_tp_f += tp_f
        total_pred_f += n_p_f
        total_gt_f += n_g_f
        all_matched_ious_f.extend(matched_ious_f)
        per_img_recall_f.append(tp_f / n_g_f if n_g_f > 0 else 1.0)

    def _summarize(tp, pr, gt, ious, per_rec, mask_cnt, fg_iou, fg_prec, fg_rec):
        prec = tp / pr if pr > 0 else 0.0
        rec = tp / gt if gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        miou = np.mean(ious) if ious else 0.0
        ar = np.mean(per_rec) if per_rec else 0.0
        avg_n = mask_cnt / valid_img_count if valid_img_count > 0 else 0.0
        return prec, rec, f1, miou, ar, avg_n, \
               fg_iou / valid_img_count if valid_img_count > 0 else 0.0, \
               fg_prec / valid_img_count if valid_img_count > 0 else 0.0, \
               fg_rec / valid_img_count if valid_img_count > 0 else 0.0, \
               tp, pr, gt

    ru = _summarize(total_tp, total_pred, total_gt, all_matched_ious, per_img_recall,
                    total_mask_count, total_fg_iou, total_fg_prec, total_fg_rec)
    rf = _summarize(total_tp_f, total_pred_f, total_gt_f, all_matched_ious_f, per_img_recall_f,
                    total_mask_count_f, total_fg_iou_f, total_fg_prec_f, total_fg_rec_f)

    avg_mask_count = total_mask_count / valid_img_count if valid_img_count > 0 else 0.0

    # ---- save preds ----
    with open(cfg["output_json"], "w") as f:
        json.dump(all_coco_preds, f)

    # quick RLE complexity check
    rle_bytes_list = []
    for p in all_coco_preds:
        c = p["segmentation"]["counts"]
        rle_bytes_list.append(len(c) if isinstance(c, str) else len(str(c)))
    if rle_bytes_list:
        print(f"  RLE complexity: {np.mean(rle_bytes_list):.0f} avg chars/pred  (max={np.max(rle_bytes_list)})")

    # ---- final summary ----
    print(f"\n{'='*50}")
    print("COCO 2017 val — FastSAM Everything-Mode Results")
    print(f"{'='*50}")
    print(f"  Valid images:      {valid_img_count}  (fail: {fail_count}, no_det: {no_det_count})")
    print(f"  Avg masks (all): {ru[5]:.1f}   (≥{cfg['min_area']}px): {rf[5]:.1f}")
    print(f"\n  ---- Instance-Level Metrics (Hungarian, IoU≥{cfg['match_iou_thresh']}) ----")
    print(f"  {'':>20} {'All masks':>16} {'≥{0}px'.format(cfg['min_area']):>16}")
    print(f"  {'Precision:':>20} {ru[0]:>16.4f} {rf[0]:>16.4f}")
    print(f"  {'Recall:':>20} {ru[1]:>16.4f} {rf[1]:>16.4f}")
    print(f"  {'F-Score:':>20} {ru[2]:>16.4f} {rf[2]:>16.4f}")
    print(f"  {'Instance mIoU:':>20} {ru[3]:>16.4f} {rf[3]:>16.4f}")
    print(f"  {'AR@300:':>20} {ru[4]:>16.4f} {rf[4]:>16.4f}")
    print(f"  {'TP/Pred/GT:':>20} {ru[9]}/{ru[10]}/{ru[11]:>16} {rf[9]}/{rf[10]}/{rf[11]}")
    print(f"\n  ---- Pixel-Level Foreground Metrics ----")
    print(f"  {'':>20} {'All masks':>16} {'≥{0}px'.format(cfg['min_area']):>16}")
    print(f"  {'Foreground IoU:':>20} {ru[6]:>16.4f} {rf[6]:>16.4f}")
    print(f"  {'Pixel Precision:':>20} {ru[7]:>16.4f} {rf[7]:>16.4f}")
    print(f"  {'Pixel Recall:':>20} {ru[8]:>16.4f} {rf[8]:>16.4f}")
    print(f"\n  Preds saved:       {cfg['output_json']}  ({len(all_coco_preds)} entries)")
    print(f"{'='*50}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit images for quick test, 0 = all")
    args = parser.parse_args()
    main(max_images=args.max_images)
