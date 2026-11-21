"""Train VL-P3 Supervised Spatial Heads (PatchSemanticHead, A1 Ablations 1..5, B1, B2).

Strictly uses `train_v3` images only for parameter fitting and evaluates controlled
training ablations + segmentation models on `val_cal_v3` (never touching `final_holdout_v3`).
Exports all serialized models and SHA-256 checksums to `models/` + `manifests/training_metrics.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .common_utils import (
    VL_P3_DIR,
    axial_angle_diff_deg,
    compute_endpoint_and_centerline_metrics,
    compute_mask_metrics,
    compute_rotated_obb_iou,
    compute_sha256,
    decode_mask_rle,
    load_image_unicode,
    obb_from_endpoints,
)
from .keypoint_detector import (
    ABLATION_MODES,
    BASE_CTX_DIM,
    SupervisedKeypointCenterlineDetector,
    extract_pixel_context_112,
)
from .shaft_segmenter import KP_GUIDANCE_DIM, SupervisedShaftSegmenter, build_kp_guidance_112
from .spatial_features import (
    DINOv2SpatialBackbone,
    GRID_SIZE,
    PatchSemanticHead,
    SpatialFeatureBundle,
    YOLOXContextExtractor,
    augment_brightness_contrast_and_occlusion,
    augment_horizontal_flip,
    augment_rotation_scale,
    build_supervision_targets_112,
    extract_spatial_features,
    pad_image_to_square_with_transform,
)


def _sample_image_pixels(
    ctx112: np.ndarray,
    targets: dict[str, np.ndarray] | None,
    valid112: np.ndarray,
    is_positive: bool,
    rng: np.random.Generator,
    ridge112: np.ndarray | None = None,
    hard_mine_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample informative positive shaft pixels + hard negative pixels from one 112x112 image."""
    h_w = GRID_SIZE * GRID_SIZE
    X_flat = ctx112.reshape(h_w, -1)
    valid_flat = valid112.reshape(h_w) > 0.4

    if is_positive and targets is not None:
        Y_flat = np.stack(
            [targets["grip_hm"], targets["tip_hm"], targets["centerline_hm"], targets["cos2"], targets["sin2"]],
            axis=-1,
        ).reshape(h_w, 5)
        fg_mask = valid_flat & (
            (Y_flat[:, 2] >= 0.08) | (Y_flat[:, 0] >= 0.08) | (Y_flat[:, 1] >= 0.08)
        )
        bg_mask = valid_flat & (~fg_mask)
        fg_idx = np.flatnonzero(fg_mask)
        bg_all = np.flatnonzero(bg_mask)

        # Keep ALL foreground & near-centerline pixels + prioritize high-ridge hard background pixels
        if ridge112 is not None and len(bg_all) > 180:
            r_bg = ridge112.ravel()[bg_all]
            n_hard = min(90, len(bg_all) // 2)
            hard_pos = np.argpartition(r_bg, -n_hard)[-n_hard:]
            hard_idx = bg_all[hard_pos]
            rand_idx = rng.choice(bg_all, size=min(140, len(bg_all)), replace=False)
            bg_idx = np.unique(np.concatenate([hard_idx, rand_idx]))
        else:
            bg_idx = rng.choice(bg_all, size=min(200, len(bg_all)), replace=False) if len(bg_all) else bg_all

        chosen = np.concatenate([fg_idx, bg_idx])
        # Higher weight on genuine cane peaks & centerline pixels
        w_fg = 3.2 + 5.0 * np.maximum.reduce([Y_flat[fg_idx, 0], Y_flat[fg_idx, 1], Y_flat[fg_idx, 2]])
        w_bg = np.full(len(bg_idx), 1.0, dtype=np.float32)
        weights = np.concatenate([w_fg.astype(np.float32), w_bg])
        return X_flat[chosen], Y_flat[chosen], weights

    # Hard negative image: all targets are zero; emphasize false high-ridge or high-scoring pixels
    all_v = np.flatnonzero(valid_flat)
    if len(all_v) == 0:
        return np.zeros((0, X_flat.shape[1]), dtype=np.float32), np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    score_map = hard_mine_scores if hard_mine_scores is not None else ridge112
    if score_map is not None and len(all_v) > 220:
        s_v = score_map.ravel()[all_v]
        n_hard = min(140, len(all_v) // 2)
        hard_idx = all_v[np.argpartition(s_v, -n_hard)[-n_hard:]]
        rand_idx = rng.choice(all_v, size=min(120, len(all_v)), replace=False)
        chosen = np.unique(np.concatenate([hard_idx, rand_idx]))
        w_mult = 2.8 if hard_mine_scores is not None else 1.5
        w = np.where(score_map.ravel()[chosen] >= 0.20, w_mult, 1.0).astype(np.float32)
    else:
        chosen = rng.choice(all_v, size=min(220, len(all_v)), replace=False)
        w = np.ones(len(chosen), dtype=np.float32)

    Y_neg = np.zeros((len(chosen), 5), dtype=np.float32)
    return X_flat[chosen], Y_neg, w


def fit_patch_semantic_head(
    train_bundles: list[tuple[dict, SpatialFeatureBundle, dict | None]],
) -> PatchSemanticHead:
    """Fit PCA(12) + 5-class logistic patch head on train_v3 DINOv2 16x16x384 tokens."""
    all_tokens = []
    sem_targets = []  # [cane_shaft, cane_end, person, confuser_neg, road_bg]

    for item, bundle, gt in train_bundles:
        tok = bundle.patches_16.reshape(256, 384)
        all_tokens.append(tok)
        f16_ridge = cv2.resize(bundle.features_112[:, :, 6], (16, 16), interpolation=cv2.INTER_AREA).ravel()
        f16_person = cv2.resize(bundle.features_112[:, :, 33], (16, 16), interpolation=cv2.INTER_AREA).ravel() > 0.25

        y_mat = np.zeros((256, 5), dtype=int)
        y_mat[:, 2] = f16_person.astype(int)

        if item["label"] == "object_01" and gt is not None:
            m = decode_mask_rle(gt["mask_rle"])
            m16 = bundle.transform.warp_mask_to_grid(m, 16).ravel()
            y_mat[:, 0] = (m16 >= 0.08).astype(int)
            ep = gt["endpoints_xyxy"]
            gx1, gy1 = bundle.transform.orig_to_grid(ep[0], ep[1])
            gx2, gy2 = bundle.transform.orig_to_grid(ep[2], ep[3])
            px1, py1 = int(np.clip(gx1 * 16.0 / GRID_SIZE, 0, 15)), int(np.clip(gy1 * 16.0 / GRID_SIZE, 0, 15))
            px2, py2 = int(np.clip(gx2 * 16.0 / GRID_SIZE, 0, 15)), int(np.clip(gy2 * 16.0 / GRID_SIZE, 0, 15))
            y_mat[py1 * 16 + px1, 1] = 1
            y_mat[py2 * 16 + px2, 1] = 1
            y_mat[:, 4] = ((y_mat[:, 0] == 0) & (~f16_person)).astype(int)
        else:
            # High-ridge patches in negative images are hard confusers; others are road/bg
            y_mat[:, 3] = (f16_ridge >= 0.24).astype(int)
            y_mat[:, 4] = (f16_ridge < 0.24).astype(int)
        sem_targets.append(y_mat)

    X_tok = np.concatenate(all_tokens, axis=0).astype(np.float32)
    Y_sem = np.concatenate(sem_targets, axis=0)

    rng = np.random.default_rng(42)
    pca_idx = rng.choice(len(X_tok), size=min(18_000, len(X_tok)), replace=False)
    pca = PCA(n_components=12, random_state=42).fit(X_tok[pca_idx])
    pca_mean = pca.mean_.astype(np.float32)
    pca_comp = pca.components_.T.astype(np.float32)  # [384, 12]

    sem_W = np.zeros((384, 5), dtype=np.float32)
    sem_b = np.zeros((5,), dtype=np.float32)
    cls_idx = rng.choice(len(X_tok), size=min(24_000, len(X_tok)), replace=False)
    Xc, Yc = X_tok[cls_idx], Y_sem[cls_idx]
    for ci in range(5):
        y_ci = Yc[:, ci]
        if int(np.sum(y_ci == 1)) >= 8 and int(np.sum(y_ci == 0)) >= 8:
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=300, random_state=42 + ci)
            clf.fit(Xc, y_ci)
            sem_W[:, ci] = clf.coef_[0].astype(np.float32)
            sem_b[ci] = float(clf.intercept_[0])

    return PatchSemanticHead(pca_mean=pca_mean, pca_components=pca_comp, sem_W=sem_W, sem_b=sem_b)


def evaluate_localization_on_val(
    detector: SupervisedKeypointCenterlineDetector,
    val_pos_items: list[tuple[dict, SpatialFeatureBundle, dict]],
) -> dict:
    """Measure keypoint, centerline, angle & rotated OBB IoU metrics on val_cal_v3."""
    rows = []
    for item, bundle, gt in val_pos_items:
        pred = detector.predict(bundle)
        h, w = item["height"], item["width"]
        gt_ep = [float(v) for v in gt["endpoints_xyxy"]]
        if gt_ep[1] > gt_ep[3]:
            gt_ep = [gt_ep[2], gt_ep[3], gt_ep[0], gt_ep[1]]
        m = compute_endpoint_and_centerline_metrics(
            pred.endpoints_xyxy,
            gt_ep,
            pred.centerline_points_xy,
            gt["centerline_points_xy"],
            (h, w),
        )
        gt_len = max(10.0, math.hypot(gt_ep[2] - gt_ep[0], gt_ep[3] - gt_ep[1]))
        gt_obb, _ = obb_from_endpoints(gt_ep[0], gt_ep[1], gt_ep[2], gt_ep[3], thickness=max(8.0, 0.028 * gt_len))
        p_obb, _ = obb_from_endpoints(pred.endpoints_xyxy[0], pred.endpoints_xyxy[1], pred.endpoints_xyxy[2], pred.endpoints_xyxy[3], thickness=max(8.0, 0.028 * gt_len))
        rot_iou = compute_rotated_obb_iou(p_obb, gt_obb)
        ang_err = axial_angle_diff_deg(
            math.degrees(math.atan2(pred.endpoints_xyxy[3] - pred.endpoints_xyxy[1], pred.endpoints_xyxy[2] - pred.endpoints_xyxy[0])),
            math.degrees(math.atan2(gt_ep[3] - gt_ep[1], gt_ep[2] - gt_ep[0])),
        )
        m["rot_obb_iou"] = round(float(rot_iou), 4)
        m["angle_error_deg"] = round(float(ang_err), 2)
        m["decode_ms"] = pred.decode_ms
        rows.append(m)

    if not rows:
        return {}
    agg = {}
    for k in rows[0].keys():
        agg[f"mean_{k}"] = round(float(np.mean([r[k] for r in rows])), 4)
    return agg


def evaluate_segmentation_on_val(
    segmenter: SupervisedShaftSegmenter,
    val_pos_items: list[tuple[dict, SpatialFeatureBundle, dict]],
    detector: SupervisedKeypointCenterlineDetector | None = None,
    mask_thr: float | None = None,
) -> dict:
    ious, f1s, precs, recs, bgs, bf1s = [], [], [], [], [], []
    for item, bundle, gt in val_pos_items:
        loc_pred = detector.predict(bundle) if (detector is not None and segmenter.mode == "kp_guided") else None
        sp = segmenter.predict(bundle, loc_pred=loc_pred, mask_thr=mask_thr)
        gt_m = decode_mask_rle(gt["mask_rle"]) > 0
        mm = compute_mask_metrics(sp.mask_orig, gt_m)
        ious.append(mm["mask_iou"])
        f1s.append(mm["dice"])
        precs.append(mm["precision"])
        recs.append(mm["recall"])
        bgs.append(mm["bg_contamination"])
        bf1s.append(mm["boundary_f1"])
    return {
        "n_eval": len(ious),
        "mean_mask_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "mean_dice_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "mean_pixel_precision": round(float(np.mean(precs)), 4) if precs else 0.0,
        "mean_pixel_recall": round(float(np.mean(recs)), 4) if recs else 0.0,
        "mean_boundary_f1": round(float(np.mean(bf1s)), 4) if bf1s else 0.0,
        "mean_bg_contamination": round(float(np.mean(bgs)), 4) if bgs else 0.0,
    }


def _collect_training_samples(
    train_items: list[tuple[dict, SpatialFeatureBundle, dict | None, np.ndarray]],
    dino: DINOv2SpatialBackbone,
    yolox: YOLOXContextExtractor,
    patch_head: PatchSemanticHead,
    use_hard_mining: bool,
    use_full_aug: bool,
    mine_detector: SupervisedKeypointCenterlineDetector | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(777)
    Xs, Ys, Ws = [], [], []

    for item, bundle, gt, img in train_items:
        is_pos = (item["label"] == "object_01" and gt is not None)
        if is_pos:
            gt_mask = decode_mask_rle(gt["mask_rle"])
            ep = [float(v) for v in gt["endpoints_xyxy"]]
            tgt = build_supervision_targets_112(ep, gt_mask, bundle.transform)
            ctx = extract_pixel_context_112(bundle.features_112)
            xi, yi, wi = _sample_image_pixels(
                ctx, tgt, bundle.features_112[:, :, 15], True, rng, ridge112=bundle.features_112[:, :, 6]
            )
            Xs.append(xi); Ys.append(yi); Ws.append(wi)

            if use_full_aug:
                # 1. Horizontal flip (reuses cheap YOLOX skip=False)
                img_f, ep_f, m_f = augment_horizontal_flip(img, ep, gt_mask)
                if ep_f is not None:
                    b_f = extract_spatial_features(img_f, dino, yolox_ctx=yolox, patch_head=patch_head)
                    t_f = build_supervision_targets_112(ep_f, m_f, b_f.transform)
                    xi, yi, wi = _sample_image_pixels(extract_pixel_context_112(b_f.features_112), t_f, b_f.features_112[:, :, 15], True, rng, ridge112=b_f.features_112[:, :, 6])
                    Xs.append(xi); Ys.append(yi); Ws.append(wi * 0.9)

                # 2. Rotation/scale + simulated mid-shaft occlusion
                ang = float(rng.uniform(-13.0, 13.0))
                sc = float(rng.uniform(0.90, 1.07))
                img_r, ep_r, m_r = augment_rotation_scale(img, ep, gt_mask, ang, sc)
                img_o, ep_o, m_o = augment_brightness_contrast_and_occlusion(img_r, ep_r, m_r, rng, apply_occlusion=True)
                if ep_o is not None:
                    b_o = extract_spatial_features(img_o, dino, yolox_ctx=yolox, patch_head=patch_head)
                    # Note: build_supervision_targets_112 preserves full endpoints & centerline even on occluded visible m_o!
                    t_o = build_supervision_targets_112(ep_o, m_o, b_o.transform)
                    xi, yi, wi = _sample_image_pixels(extract_pixel_context_112(b_o.features_112), t_o, b_o.features_112[:, :, 15], True, rng, ridge112=b_o.features_112[:, :, 6])
                    Xs.append(xi); Ys.append(yi); Ws.append(wi * 1.1)
        else:
            ctx = extract_pixel_context_112(bundle.features_112)
            mine_sc = None
            if use_hard_mining and mine_detector is not None:
                gh, th, ch, _, _, _ = mine_detector.predict_heatmaps_112(bundle.features_112)
                mine_sc = np.maximum.reduce([gh, th, ch])
            xi, yi, wi = _sample_image_pixels(
                ctx, None, bundle.features_112[:, :, 15], False, rng,
                ridge112=bundle.features_112[:, :, 6], hard_mine_scores=mine_sc,
            )
            Xs.append(xi); Ys.append(yi); Ws.append(wi)

    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), np.concatenate(Ws, axis=0)


def train_vl_p3_models() -> Path:
    t_start = time.perf_counter()
    manifests_dir = VL_P3_DIR / "manifests"
    models_dir = VL_P3_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ds = json.loads((manifests_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    ann = json.loads((manifests_dir / "annotation_manifest.json").read_text(encoding="utf-8"))["annotations"]

    dino = DINOv2SpatialBackbone(models_dir / "dinov2_small_quantized.onnx")
    yolox = YOLOXContextExtractor(models_dir / "yolox_nano.onnx")

    def _load_split(split_name: str, head: PatchSemanticHead | None = None):
        out = []
        for item in ds[split_name]:
            img = load_image_unicode(Path(item["abs_path"]))
            gt = ann.get(item["id"])
            bundle = extract_spatial_features(img, dino, yolox_ctx=yolox, patch_head=head)
            out.append((item, bundle, gt, img))
        return out

    print("Extracting base features on train_v3...", flush=True)
    raw_train = _load_split("train_v3", head=None)
    print(f"Loaded {len(raw_train)} train_v3 items; fitting PatchSemanticHead...", flush=True)
    patch_head = fit_patch_semantic_head([(it, b, gt) for it, b, gt, _ in raw_train])
    patch_head_path = models_dir / "patch_semantic_head.json"
    patch_head_path.write_text(json.dumps(patch_head.to_state_dict(), indent=2), encoding="utf-8")

    # Refresh pca112 & sem112 channels in-place with trained patch_head (0 extra ONNX calls!)
    def _refresh_patch_channels(records: list[tuple[dict, SpatialFeatureBundle, dict | None, np.ndarray]]):
        for _, bundle, _, _ in records:
            pca112, sem112 = patch_head.project_grid_to_112(bundle.patches_16)
            bundle.features_112[:, :, 16:28] = pca112
            bundle.features_112[:, :, 28:33] = sem112

    _refresh_patch_channels(raw_train)
    raw_val = _load_split("val_cal_v3", head=patch_head)
    val_pos = [(it, b, gt) for it, b, gt, _ in raw_val if it["label"] == "object_01" and gt is not None]
    print(f"Loaded {len(raw_val)} val_cal_v3 items ({len(val_pos)} object_01 GT).", flush=True)

    # Execute all 5 controlled training ablations on val_cal_v3
    ablation_results: dict[str, dict] = {}
    base_X, base_Y, base_W = _collect_training_samples(raw_train, dino, yolox, patch_head, use_hard_mining=False, use_full_aug=False)

    stage3_detector: SupervisedKeypointCenterlineDetector | None = None
    best_detector: SupervisedKeypointCenterlineDetector | None = None

    for mode in ABLATION_MODES:
        t_m = time.perf_counter()
        if mode in ("1_keypoints_only", "2_centerline_only", "3_keypoints_plus_centerline"):
            X_use, Y_use, W_use = base_X, base_Y, base_W
        elif mode == "4_kp_cl_plus_hard_neg_mining":
            X_use, Y_use, W_use = _collect_training_samples(
                raw_train, dino, yolox, patch_head, use_hard_mining=True, use_full_aug=False, mine_detector=stage3_detector
            )
        else:
            X_use, Y_use, W_use = _collect_training_samples(
                raw_train, dino, yolox, patch_head, use_hard_mining=True, use_full_aug=True, mine_detector=stage3_detector
            )

        det = SupervisedKeypointCenterlineDetector(mode=mode).fit(X_use, Y_use, sample_weights=W_use)
        val_m = evaluate_localization_on_val(det, val_pos)
        det.train_summary = {"mode": mode, "n_pixel_samples": int(len(X_use)), "val_cal_v3_metrics": val_m}
        ablation_results[mode] = {
            "n_pixel_samples": int(len(X_use)),
            "train_sec": round(time.perf_counter() - t_m, 2),
            "val_cal_v3": val_m,
        }
        print(f"Ablation [{mode}] done ({len(X_use)} px) -> {val_m}", flush=True)
        if mode == "3_keypoints_plus_centerline":
            stage3_detector = det
        if mode == "5_full_aug_and_occlusion":
            best_detector = det

    assert best_detector is not None
    kp_model_path = models_dir / "keypoint_detector_v3.joblib"
    kp_sha = best_detector.save(kp_model_path)

    # Now train B1 (standalone) and B2 (keypoint-guided) shaft segmenters on train_v3 (+ aug/occlusion + jitter)
    rng = np.random.default_rng(909)
    b1_Xs, b1_Ys, b1_Ws = [], [], []
    b2_Xs, b2_Ys, b2_Ws = [], [], []

    def _jitter_loc(bundle: SpatialFeatureBundle, ep: list[float], rng_l: np.random.Generator):
        # Mix real predicted A1 (70%) or lightly jittered GT (30%) during train so B2 tolerates slight endpoint shift
        pred_loc = best_detector.predict(bundle)
        if rng_l.random() < 0.35:
            jep = [float(v + rng_l.uniform(-12.0, 12.0)) for v in ep]
            if jep[1] > jep[3]:
                jep = [jep[2], jep[3], jep[0], jep[1]]
            tgt = build_supervision_targets_112(jep, None, bundle.transform)
            pred_loc.endpoints_xyxy = jep
            pred_loc.cl_hm = tgt["centerline_hm"]
            pred_loc.grip_hm = tgt["grip_hm"]
            pred_loc.tip_hm = tgt["tip_hm"]
            pred_loc.detected = True
        return pred_loc

    for item, bundle, gt, img in raw_train:
        is_pos = (item["label"] == "object_01" and gt is not None)
        if is_pos:
            gt_mask = decode_mask_rle(gt["mask_rle"])
            ep = [float(v) for v in gt["endpoints_xyxy"]]
            variants = [(img, ep, gt_mask, bundle, 1.0)]
            img_f, ep_f, m_f = augment_horizontal_flip(img, ep, gt_mask)
            if ep_f is not None:
                variants.append((img_f, ep_f, m_f, extract_spatial_features(img_f, dino, yolox, patch_head), 0.9))
            ang = float(rng.uniform(-12.0, 12.0))
            img_r, ep_r, m_r = augment_rotation_scale(img, ep, gt_mask, ang, float(rng.uniform(0.92, 1.06)))
            img_o, ep_o, m_o = augment_brightness_contrast_and_occlusion(img_r, ep_r, m_r, rng, apply_occlusion=True)
            if ep_o is not None:
                variants.append((img_o, ep_o, m_o, extract_spatial_features(img_o, dino, yolox, patch_head), 1.05))

            for _, v_ep, v_mask, v_bnd, w_s in variants:
                v_tgt = build_supervision_targets_112(v_ep, v_mask, v_bnd.transform)
                ctx = extract_pixel_context_112(v_bnd.features_112)
                y_m = (v_tgt["shaft_mask"].ravel() >= 0.35).astype(np.float32)
                valid_f = v_bnd.features_112[:, :, 15].ravel() > 0.4
                fg = np.flatnonzero(valid_f & (y_m > 0.5))
                bg_all = np.flatnonzero(valid_f & (y_m < 0.5))
                bg_r = v_bnd.features_112[:, :, 6].ravel()[bg_all]
                hard_b = bg_all[np.argpartition(bg_r, -min(80, len(bg_r)))[-min(80, len(bg_r)):]] if len(bg_r) > 80 else bg_all
                rand_b = rng.choice(bg_all, size=min(120, len(bg_all)), replace=False) if len(bg_all) else bg_all
                ch = np.concatenate([fg, np.unique(np.concatenate([hard_b, rand_b]))])
                w_arr = np.concatenate([np.full(len(fg), 3.6 * w_s, dtype=np.float32), np.full(len(ch) - len(fg), 1.0 * w_s, dtype=np.float32)])

                b1_Xs.append(ctx.reshape(-1, BASE_CTX_DIM)[ch])
                b1_Ys.append(y_m[ch])
                b1_Ws.append(w_arr)

                loc_g = _jitter_loc(v_bnd, v_ep, rng)
                kpg = build_kp_guidance_112(loc_g, v_bnd)
                b2_in = np.concatenate([ctx, kpg], axis=-1).reshape(-1, BASE_CTX_DIM + KP_GUIDANCE_DIM)
                b2_Xs.append(b2_in[ch])
                b2_Ys.append(y_m[ch])
                b2_Ws.append(w_arr)
        else:
            ctx = extract_pixel_context_112(bundle.features_112)
            valid_f = bundle.features_112[:, :, 15].ravel() > 0.4
            all_v = np.flatnonzero(valid_f)
            if len(all_v) == 0:
                continue
            r_sc = bundle.features_112[:, :, 6].ravel()[all_v]
            nh = min(130, len(all_v) // 2)
            ch = np.unique(np.concatenate([all_v[np.argpartition(r_sc, -nh)[-nh:]], rng.choice(all_v, size=min(110, len(all_v)), replace=False)]))
            y_n = np.zeros(len(ch), dtype=np.float32)
            w_n = np.where(bundle.features_112[:, :, 6].ravel()[ch] >= 0.22, 2.5, 1.0).astype(np.float32)
            b1_Xs.append(ctx.reshape(-1, BASE_CTX_DIM)[ch]); b1_Ys.append(y_n); b1_Ws.append(w_n)

            loc_neg = best_detector.predict(bundle)
            kpg = build_kp_guidance_112(loc_neg, bundle)
            b2_in = np.concatenate([ctx, kpg], axis=-1).reshape(-1, BASE_CTX_DIM + KP_GUIDANCE_DIM)
            b2_Xs.append(b2_in[ch]); b2_Ys.append(y_n); b2_Ws.append(w_n)

    seg_b1 = SupervisedShaftSegmenter(mode="standalone").fit(
        np.concatenate(b1_Xs, axis=0), np.concatenate(b1_Ys, axis=0), np.concatenate(b1_Ws, axis=0)
    )
    seg_b2 = SupervisedShaftSegmenter(mode="kp_guided").fit(
        np.concatenate(b2_Xs, axis=0), np.concatenate(b2_Ys, axis=0), np.concatenate(b2_Ws, axis=0)
    )

    # Calibrate mask threshold on val_cal_v3 for B1 and B2
    for seg_obj, det_ref in ((seg_b1, None), (seg_b2, best_detector)):
        best_f1, best_thr = -1.0, 0.42
        for thr in (0.34, 0.38, 0.42, 0.46, 0.50):
            ev = evaluate_segmentation_on_val(seg_obj, val_pos, det_ref, mask_thr=thr)
            if ev["mean_dice_f1"] > best_f1:
                best_f1 = ev["mean_dice_f1"]
                best_thr = thr
        seg_obj.mask_thr = best_thr

    b1_val = evaluate_segmentation_on_val(seg_b1, val_pos, None)
    b2_val = evaluate_segmentation_on_val(seg_b2, val_pos, best_detector)
    seg_b1.train_summary = {"mode": "standalone", "mask_thr": seg_b1.mask_thr, "val_cal_v3": b1_val}
    seg_b2.train_summary = {"mode": "kp_guided", "mask_thr": seg_b2.mask_thr, "val_cal_v3": b2_val}

    b1_path = models_dir / "shaft_segmenter_b1_standalone.joblib"
    b2_path = models_dir / "shaft_segmenter_b2_kp_guided.joblib"
    b1_sha = seg_b1.save(b1_path)
    b2_sha = seg_b2.save(b2_path)

    print(f"B1 standalone val_cal_v3 -> {b1_val}", flush=True)
    print(f"B2 kp_guided val_cal_v3  -> {b2_val}", flush=True)

    summary = {
        "train_split_counts": {
            "train_v3_total": len(raw_train),
            "train_v3_pos": sum(1 for it, _, _, _ in raw_train if it["label"] == "object_01"),
            "val_cal_v3_total": len(raw_val),
            "val_cal_v3_pos": len(val_pos),
        },
        "training_ablations": ablation_results,
        "segmentation_val_cal_v3": {
            "B1_standalone": b1_val,
            "B2_kp_guided": b2_val,
        },
        "artifacts": {
            "patch_semantic_head": {"path": str(patch_head_path), "sha256": compute_sha256(patch_head_path)},
            "keypoint_detector_v3": {"path": str(kp_model_path), "sha256": kp_sha},
            "shaft_segmenter_b1": {"path": str(b1_path), "sha256": b1_sha},
            "shaft_segmenter_b2": {"path": str(b2_path), "sha256": b2_sha},
        },
        "total_train_sec": round(time.perf_counter() - t_start, 2),
    }
    out_json = manifests_dir / "training_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training metrics -> {out_json}", flush=True)
    return out_json


if __name__ == "__main__":
    train_vl_p3_models()
