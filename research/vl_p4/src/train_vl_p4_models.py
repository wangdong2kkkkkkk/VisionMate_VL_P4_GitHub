"""Train A1_CNN_Keypoint, B1_CNN_Segmentation, B2_CNN_MultiTask, export ONNX FP32/INT8 & train SoftCaneVerifierV4."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_cane_net import (
    GRID_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    MobileNetV3SmallMultiHead,
    ONNXCaneRunner,
    preprocess_image_256,
)
from .common_utils import (
    VL_P4_DIR,
    compute_endpoint_and_centerline_metrics,
    compute_mask_metrics,
    compute_sha256,
    decode_mask_rle,
    load_image_unicode,
    order_endpoints_top_bottom,
)
from .soft_verifier_v4 import SoftCaneVerifierV4, extract_soft_line_features


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _gaussian_heatmap_64(cx: float, cy: float, sigma: float = 2.2) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
    return np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (sigma ** 2))).astype(np.float32)


def _line_heatmap_64(x1: float, y1: float, x2: float, y2: float, sigma: float = 1.6) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE].astype(np.float32)
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom < 1e-6:
        return _gaussian_heatmap_64(x1, y1, sigma)
    t = np.clip(((xx - x1) * vx + (yy - y1) * vy) / denom, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    return np.exp(-0.5 * dist2 / (sigma ** 2)).astype(np.float32)


def _build_targets_64(ep_grid: list[float] | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets [4, 64, 64], has_mask [1]) in 64x64 space."""
    tgt = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if ep_grid is None:
        return tgt, np.array([0.0], dtype=np.float32)
    x1, y1, x2, y2 = ep_grid
    if (y1 > y2) or (abs(y1 - y2) <= 1e-3 and x1 > x2):
        x1, y1, x2, y2 = x2, y2, x1, y1
    tgt[0] = _gaussian_heatmap_64(x1, y1, sigma=2.2)
    tgt[1] = _gaussian_heatmap_64(x2, y2, sigma=2.2)
    tgt[2] = _line_heatmap_64(x1, y1, x2, y2, sigma=1.6)
    # Binary-like shaft mask on 64x64
    m64 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    cv2.line(m64, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 2)
    tgt[3] = (m64 > 0).astype(np.float32)
    return tgt, np.array([1.0], dtype=np.float32)


def _augment_sample(sq256: np.ndarray, ep_grid: list[float] | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    img = sq256.copy()
    eg = list(ep_grid) if ep_grid is not None else None

    # 1. Horizontal flip (p=0.5)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        if eg is not None:
            eg = [GRID_SIZE - 1.0 - eg[0], eg[1], GRID_SIZE - 1.0 - eg[2], eg[3]]

    # 2. Small affine rotation (+/-15 deg), scale (0.86..1.14), shift (+/-6%)
    ang = float(rng.uniform(-15.0, 15.0))
    sc = float(rng.uniform(0.86, 1.14))
    tx = float(rng.uniform(-0.06, 0.06)) * INPUT_SIZE
    ty = float(rng.uniform(-0.06, 0.06)) * INPUT_SIZE
    M = cv2.getRotationMatrix2D((INPUT_SIZE * 0.5, INPUT_SIZE * 0.5), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR, borderValue=(127, 127, 127))

    if eg is not None:
        # Transform endpoints from 64x64 -> 256x256 -> affine -> 64x64
        s4 = float(INPUT_SIZE) / float(GRID_SIZE)
        pts = np.array([[eg[0] * s4, eg[1] * s4, 1.0], [eg[2] * s4, eg[3] * s4, 1.0]], dtype=np.float32)
        tp = (M @ pts.T).T / s4
        eg = [
            float(np.clip(tp[0, 0], 0.0, GRID_SIZE - 1.0)),
            float(np.clip(tp[0, 1], 0.0, GRID_SIZE - 1.0)),
            float(np.clip(tp[1, 0], 0.0, GRID_SIZE - 1.0)),
            float(np.clip(tp[1, 1], 0.0, GRID_SIZE - 1.0)),
        ]

    # 3. Random occlusion cutout along image (p=0.25)
    if rng.random() < 0.25:
        cw = int(rng.integers(18, 46))
        ch = int(rng.integers(18, 46))
        cx = int(rng.integers(0, INPUT_SIZE - cw))
        cy = int(rng.integers(0, INPUT_SIZE - ch))
        img[cy : cy + ch, cx : cx + cw] = int(rng.integers(40, 180))

    # 4. Brightness & contrast jitter
    alpha = float(rng.uniform(0.82, 1.18))
    beta = float(rng.uniform(-18.0, 18.0))
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = ((rgb - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]).transpose(2, 0, 1).astype(np.float32)
    tgt, _ = _build_targets_64(eg)
    return norm, tgt, (1.0 if eg is not None else 0.0)


def _prepare_cached_split(items: list[dict], ann: dict) -> list[dict]:
    cached = []
    for it in items:
        img = load_image_unicode(Path(it["abs_path"]))
        tensor, sq256, tfm = preprocess_image_256(img)
        is_pos = it.get("label") == "object_01"
        ep_grid = None
        if is_pos and it["id"] in ann:
            ep = ann[it["id"]]["endpoints_xyxy"]
            gx1, gy1 = tfm.orig_to_grid(ep[0], ep[1])
            gx2, gy2 = tfm.orig_to_grid(ep[2], ep[3])
            ep_grid = [
                float(np.clip(gx1, 0.0, GRID_SIZE - 1.0)),
                float(np.clip(gy1, 0.0, GRID_SIZE - 1.0)),
                float(np.clip(gx2, 0.0, GRID_SIZE - 1.0)),
                float(np.clip(gy2, 0.0, GRID_SIZE - 1.0)),
            ]
        tgt, _ = _build_targets_64(ep_grid)
        cached.append({
            "id": it["id"],
            "is_pos": is_pos,
            "img": img,
            "sq256": sq256,
            "tensor": tensor[0],
            "tfm": tfm,
            "ep_grid": ep_grid,
            "tgt": tgt,
            "gt_ann": ann.get(it["id"]),
        })
    return cached


def _compute_loss(
    sp_logits: torch.Tensor,
    obj_logit: torch.Tensor,
    tgt: torch.Tensor,
    y_cls: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    obj_loss = F.binary_cross_entropy_with_logits(obj_logit.squeeze(1), y_cls)
    pos_mask = y_cls > 0.5
    if not torch.any(pos_mask):
        return obj_loss

    # Spatial heatmap/mask losses weighted cleanly so negatives also suppress false peaks
    if mode == "A1_CNN_Keypoint":
        # Channels 0,1,2: top_kp, bot_kp, centerline
        p_sp = sp_logits[:, :3]
        t_sp = tgt[:, :3]
        pw = torch.tensor([6.0, 6.0, 5.0], device=sp_logits.device).view(1, 3, 1, 1)
        bce_sp = F.binary_cross_entropy_with_logits(p_sp, t_sp, pos_weight=pw)
        prob = torch.sigmoid(p_sp)
        mse_sp = F.mse_loss(prob, t_sp) * 4.0
        return obj_loss + 1.2 * bce_sp + mse_sp

    if mode == "B1_CNN_Segmentation":
        p_m = sp_logits[:, 3:4]
        t_m = tgt[:, 3:4]
        bce_m = F.binary_cross_entropy_with_logits(p_m, t_m, pos_weight=torch.tensor([8.0], device=sp_logits.device))
        prob_m = torch.sigmoid(p_m)
        inter = (prob_m * t_m).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + 1.0) / (prob_m.sum(dim=(2, 3)) + t_m.sum(dim=(2, 3)) + 1.0)
        return obj_loss + 0.8 * bce_m + 0.8 * dice.mean()

    # B2_CNN_MultiTask: KP + CL prioritized, soft auxiliary mask
    p_kp = sp_logits[:, :3]
    t_kp = tgt[:, :3]
    pw = torch.tensor([6.0, 6.0, 5.0], device=sp_logits.device).view(1, 3, 1, 1)
    kp_loss = F.binary_cross_entropy_with_logits(p_kp, t_kp, pos_weight=pw) + 4.0 * F.mse_loss(torch.sigmoid(p_kp), t_kp)

    p_m = sp_logits[:, 3:4]
    t_m = tgt[:, 3:4]
    prob_m = torch.sigmoid(p_m)
    inter = (prob_m * t_m).sum(dim=(2, 3))
    dice_m = 1.0 - (2.0 * inter + 1.0) / (prob_m.sum(dim=(2, 3)) + t_m.sum(dim=(2, 3)) + 1.0)
    seg_loss = 0.5 * F.binary_cross_entropy_with_logits(p_m, t_m, pos_weight=torch.tensor([8.0], device=sp_logits.device)) + 0.5 * dice_m.mean()

    return obj_loss + 1.3 * kp_loss + 0.45 * seg_loss


def _eval_model_on_val(model: nn.Module, val_cached: list[dict], mode: str) -> dict:
    from .cnn_cane_net import decode_cnn_outputs

    model.eval()
    scores, labels, ep_errs, pdj10s, mask_ious = [], [], [], [], []
    with torch.no_grad():
        for rec in val_cached:
            x = torch.from_numpy(rec["tensor"][None, ...])
            sp, ob = model(x)
            dec = decode_cnn_outputs(sp.numpy(), ob.numpy(), rec["tfm"], mode=mode)
            scores.append(dec["obj_prob"])
            labels.append(1 if rec["is_pos"] else 0)
            if rec["is_pos"] and rec["gt_ann"] is not None:
                gt = rec["gt_ann"]
                em = compute_endpoint_and_centerline_metrics(
                    dec["endpoints_xyxy"], gt["endpoints_xyxy"],
                    dec["centerline_points_xy"], gt["centerline_points_xy"],
                    rec["img"].shape[:2],
                )
                ep_errs.append(em["endpoint_err_px"])
                pdj10s.append(em["keypoint_both_pdj_10"])
                if mode != "A1_CNN_Keypoint":
                    gt_m = decode_mask_rle(gt["mask_rle"]) > 0
                    mm = compute_mask_metrics(dec["mask_orig"], gt_m)
                    mask_ious.append(mm["mask_iou"])
    roc = float(roc_auc_score(labels, scores))
    pr = float(average_precision_score(labels, scores))
    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "mean_endpoint_err_px": round(float(np.mean(ep_errs)), 2) if ep_errs else 999.0,
        "mean_both_pdj_10": round(float(np.mean(pdj10s)), 4) if pdj10s else 0.0,
        "mean_mask_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else 0.0,
    }


def _train_single_variant(mode: str, tr_cached: list[dict], val_cached: list[dict], models_dir: Path, seed: int = 42) -> dict:
    t0 = time.perf_counter()
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    model = MobileNetV3SmallMultiHead(pretrained=True, fpn_ch=64)

    ckpt_path = models_dir / f"{mode.lower()}.pt"
    if ckpt_path.is_file():
        print(f"  [{mode}] Found existing checkpoint {ckpt_path.name}, skipping re-training and running ONNX export + parity...", flush=True)
        best_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best_state)
        best_val = _eval_model_on_val(model, val_cached, mode)
    else:
        pos_pool = [r for r in tr_cached if r["is_pos"]]
        neg_pool = [r for r in tr_cached if not r["is_pos"]]

        # Stage 1: Freeze backbone features, warm up FPN + heads (4 epochs)
        for p in model.features.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.8e-3, weight_decay=1e-4)

        best_score = -1e9
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_val = {}

        def _run_epoch(ep_idx: int, optimizer: torch.optim.Optimizer) -> float:
            model.train()
            p_idx = rng.integers(0, len(pos_pool), size=160)
            n_idx = rng.integers(0, len(neg_pool), size=160)
            batch_items = [pos_pool[i] for i in p_idx] + [neg_pool[i] for i in n_idx]
            rng.shuffle(batch_items)
            losses = []
            bs = 24
            for s in range(0, len(batch_items), bs):
                sub = batch_items[s : s + bs]
                xs, ts, ys = [], [], []
                for it in sub:
                    nx, nt, ny = _augment_sample(it["sq256"], it["ep_grid"], rng)
                    xs.append(nx)
                    ts.append(nt)
                    ys.append(ny)
                xb = torch.from_numpy(np.stack(xs, axis=0))
                tb = torch.from_numpy(np.stack(ts, axis=0))
                yb = torch.tensor(ys, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                sp_out, obj_out = model(xb)
                loss = _compute_loss(sp_out, obj_out, tb, yb, mode)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                losses.append(float(loss.item()))
            return float(np.mean(losses))

        for ep in range(1, 5):
            tr_l = _run_epoch(ep, opt)
            vm = _eval_model_on_val(model, val_cached, mode)
            print(f"  [{mode}] Warmup {ep}/4 loss={tr_l:.4f} val_roc={vm['roc_auc']:.4f} val_pr={vm['pr_auc']:.4f} ep_err={vm['mean_endpoint_err_px']:.1f}", flush=True)

        # Stage 2: Unfreeze all & fine-tune (12 epochs with CosineAnnealingLR)
        for p in model.features.parameters():
            p.requires_grad = True
        opt2 = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": 2.2e-4},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("features.")], "lr": 6.5e-4},
        ], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=12, eta_min=2e-5)

        for ep in range(1, 13):
            tr_l = _run_epoch(ep + 4, opt2)
            sched.step()
            vm = _eval_model_on_val(model, val_cached, mode)
            sel = vm["roc_auc"] + 0.5 * vm["pr_auc"] - 0.0012 * vm["mean_endpoint_err_px"] + 0.3 * vm["mean_both_pdj_10"]
            if sel > best_score:
                best_score = sel
                best_val = vm
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 3 == 0 or ep == 12:
                print(f"  [{mode}] FT {ep}/12 loss={tr_l:.4f} val_roc={vm['roc_auc']:.4f} val_pr={vm['pr_auc']:.4f} ep_err={vm['mean_endpoint_err_px']:.1f} pdj={vm['mean_both_pdj_10']:.3f} iou={vm['mean_mask_iou']:.3f}", flush=True)

        model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
    model.eval()

    # Export ONNX FP32 (dynamo=False legacy exporter for widest ORT INT8 compatibility)
    onnx_fp32 = models_dir / f"{mode.lower()}_fp32.onnx"
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_fp32),
        input_names=["input"],
        output_names=["spatial_logits", "obj_logit"],
        opset_version=17,
        dynamo=False,
    )

    # Dynamic INT8 quantization & parity check on val_cal_v4 (use relpath to avoid C++ fopen Unicode issues)
    import os
    onnx_int8 = models_dir / f"{mode.lower()}_int8.onnx"
    quantize_dynamic(os.path.relpath(onnx_fp32), os.path.relpath(onnx_int8), weight_type=QuantType.QUInt8)

    r_fp32 = ONNXCaneRunner(onnx_fp32, mode=mode)
    r_int8 = ONNXCaneRunner(onnx_int8, mode=mode)
    fp32_scores, int8_scores, labels = [], [], []
    for rec in val_cached:
        o1 = r_fp32.predict(rec["img"])
        o2 = r_int8.predict(rec["img"])
        fp32_scores.append(o1["obj_prob"])
        int8_scores.append(o2["obj_prob"])
        labels.append(1 if rec["is_pos"] else 0)

    roc_fp32 = float(roc_auc_score(labels, fp32_scores))
    roc_int8 = float(roc_auc_score(labels, int8_scores))
    max_prob_diff = float(np.max(np.abs(np.array(fp32_scores) - np.array(int8_scores))))
    int8_viable = (roc_fp32 - roc_int8) <= 0.025

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "mode": mode,
        "seed": seed,
        "train_sec": round(time.perf_counter() - t0, 2),
        "n_params": int(n_params),
        "fp32_mb": round(onnx_fp32.stat().st_size / (1024.0 * 1024.0), 3),
        "int8_mb": round(onnx_int8.stat().st_size / (1024.0 * 1024.0), 3),
        "fp32_sha256": compute_sha256(onnx_fp32),
        "int8_sha256": compute_sha256(onnx_int8),
        "best_val_cal_v4": best_val,
        "onnx_parity": {
            "val_roc_fp32": round(roc_fp32, 4),
            "val_roc_int8": round(roc_int8, 4),
            "max_prob_diff": round(max_prob_diff, 4),
            "int8_viable": bool(int8_viable),
            "recommended_onnx": onnx_int8.name if int8_viable else onnx_fp32.name,
        },
    }


def train_all_vl_p4() -> Path:
    manifests_dir = VL_P4_DIR / "manifests"
    models_dir = VL_P4_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ds = json.loads((manifests_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((manifests_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]

    print("Pre-caching train_v4 & val_cal_v4 256x256 tensors...", flush=True)
    tr_cached = _prepare_cached_split(ds["train_v4"], ann)
    val_cached = _prepare_cached_split(ds["val_cal_v4"], ann)
    print(f"Cached train_v4={len(tr_cached)} ({sum(r['is_pos'] for r in tr_cached)} pos), val_cal_v4={len(val_cached)} ({sum(r['is_pos'] for r in val_cached)} pos)", flush=True)

    results = {}
    for mode in ("A1_CNN_Keypoint", "B1_CNN_Segmentation", "B2_CNN_MultiTask"):
        print(f"\n=== Training {mode} ===", flush=True)
        results[mode] = _train_single_variant(mode, tr_cached, val_cached, models_dir, seed=42)

    # Select best CNN candidate on val_cal_v4 (between A1_CNN_Keypoint and B2_CNN_MultiTask) to wrap with C1 SoftVerifier
    best_mode = max(
        ("A1_CNN_Keypoint", "B2_CNN_MultiTask"),
        key=lambda m: results[m]["best_val_cal_v4"]["roc_auc"] + 0.4 * results[m]["best_val_cal_v4"]["pr_auc"],
    )
    rec_onnx = models_dir / results[best_mode]["onnx_parity"]["recommended_onnx"]
    print(f"\n=== Training SoftCaneVerifierV4 on {best_mode} ({rec_onnx.name}) ===", flush=True)
    runner = ONNXCaneRunner(rec_onnx, mode=best_mode)
    X_tr, y_tr = [], []
    for rec in tr_cached:
        pred = runner.predict(rec["img"])
        vec, _ = extract_soft_line_features(rec["img"], pred)
        X_tr.append(vec)
        y_tr.append(1 if rec["is_pos"] else 0)
    ver = SoftCaneVerifierV4()
    oof = ver.fit(np.stack(X_tr, axis=0), np.asarray(y_tr, dtype=np.int32), seed=42)
    ver_path = models_dir / "soft_verifier_v4.joblib"
    ver.save(ver_path)
    print(f"SoftCaneVerifierV4 OOF: {oof}", flush=True)

    out_path = manifests_dir / "training_metrics.json"
    out_path.write_text(json.dumps({
        "seed": 42,
        "input_size": INPUT_SIZE,
        "grid_size": GRID_SIZE,
        "backbone": "mobilenet_v3_small (ImageNet DEFAULT, BSD-3-Clause)",
        "variants": results,
        "verifier_base_mode": best_mode,
        "verifier_base_onnx": rec_onnx.name,
        "verifier_oof": oof,
        "verifier_sha256": compute_sha256(ver_path),
    }, indent=2), encoding="utf-8")
    print(f"Saved training metrics -> {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    train_all_vl_p4()
