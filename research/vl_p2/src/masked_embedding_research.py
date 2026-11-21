"""Method B2: Six Cane Input Representations + Masked DINOv2-Small INT8 Embedding Engine.

Compares 6 crop/mask input transformations for frozen DINOv2-Small INT8 without changing the model:
  1. rep1_axis_crop: original axis-aligned bounding box padded to square
  2. rep2_rotated_crop: rectified oriented cane corridor crop rotated horizontal, padded to square
  3. rep3_masked_crop: axis-aligned crop with predicted cane mask removing background clutter
  4. rep4_rotated_masked_crop: rotated corridor crop + predicted shaft mask suppressing background
  5. rep5_adaptive_pad_crop: rotated crop with adaptive tight corridor height (2.4x shaft thickness)
  6. rep6_minimal_corridor_crop: multi-strip longitudinal tiling of the rectified masked/unmasked cane
     [upper handle/mid, mid shaft, lower/red tip] stacked vertically into 224x224 so DINOv2 ViT
     patches see high-resolution shaft geometry and red-tip transitions (>40% foreground tokens).

Crucially aligns query and enrolled `train_pos` prototype representations so prototypes of
`object_01` are extracted using the exact same representation function as query candidates.
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .common_utils import DINOV2_SHA256, OBJECT_IDS, compute_sha256, load_image_unicode, obb_from_endpoints, pad_to_square
from .localization_research import OrientedCaneLocalizer, OrientedCaneProposal
from .segmentation_research import CaneShaftSegmenter, SegmentationResult

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REPRESENTATION_NAMES = (
    "rep1_axis_crop",
    "rep2_rotated_crop",
    "rep3_masked_crop",
    "rep4_rotated_masked_crop",
    "rep5_adaptive_pad_crop",
    "rep6_minimal_corridor_crop",
)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        out = np.zeros_like(v, dtype=np.float32)
        if out.size > 0:
            out[0] = 1.0
        return out
    return (v / n).astype(np.float32)


def _neutral_imagenet_bgr() -> tuple[int, int, int]:
    # ImageNet mean in RGB -> (0.485, 0.456, 0.406) -> BGR (104, 116, 124)
    return (104, 116, 124)


def _extract_axis_crop(image: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox_xywh]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + max(1, bw))
    y2 = min(h, y + max(1, bh))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def build_representation_canvas(
    image: np.ndarray,
    proposal: OrientedCaneProposal,
    seg_res: SegmentationResult,
    rep_name: str,
    target_size: int = 224,
) -> np.ndarray:
    """Render one of the 6 candidate representations as a 224x224 BGR image."""
    if rep_name not in REPRESENTATION_NAMES:
        raise ValueError(f"Unknown representation: {rep_name}")

    bg_bgr = _neutral_imagenet_bgr()
    h, w = image.shape[:2]
    ax, ay, aw, ah = proposal.bbox_xywh
    x1, y1 = max(0, int(ax)), max(0, int(ay))
    x2, y2 = min(w, int(ax + max(1, aw))), min(h, int(ay + max(1, ah)))

    if rep_name == "rep1_axis_crop":
        crop = _extract_axis_crop(image, proposal.bbox_xywh)
        border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
        med = tuple(int(c) for c in np.median(border, axis=0))
        sq = pad_to_square(crop, pad_color=med)
        return cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_AREA if min(sq.shape[:2]) > target_size else cv2.INTER_LINEAR)

    if rep_name == "rep2_rotated_crop":
        strip = seg_res.strip_bgr
        sq = pad_to_square(strip, pad_color=bg_bgr)
        return cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    if rep_name == "rep3_masked_crop":
        crop_u8 = image[y1:y2, x1:x2]
        ch, cw = crop_u8.shape[:2]
        s_box = float(target_size) / float(max(ch, cw, 1))
        nw = max(1, int(round(cw * s_box)))
        nh = max(1, int(round(ch * s_box)))
        crop = cv2.resize(crop_u8, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if seg_res.full_mask_soft is not None:
            alpha_2d = cv2.resize(
                seg_res.full_mask_soft[y1:y2, x1:x2],
                (nw, nh),
                interpolation=cv2.INTER_LINEAR,
            )
        elif seg_res.strip_mask_soft is not None and seg_res.affine_inv is not None:
            m_inv_crop = seg_res.affine_inv.copy()
            m_inv_crop[0, 2] -= float(x1)
            m_inv_crop[1, 2] -= float(y1)
            m_inv_crop[:2, :] *= s_box
            alpha_2d = cv2.warpAffine(
                seg_res.strip_mask_soft,
                m_inv_crop,
                (nw, nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        else:
            alpha_2d = np.ones((nh, nw), dtype=np.float32)
        alpha_eff = np.clip(alpha_2d[:, :, None] * 0.96 + 0.04, 0.0, 1.0)
        blur_bg = cv2.GaussianBlur(crop, (15, 15), 5.0).astype(np.float32)
        bg = 0.15 * blur_bg + 0.85 * np.full_like(crop, np.array(bg_bgr, dtype=np.float32))
        masked = np.clip(crop * alpha_eff + bg * (1.0 - alpha_eff), 0, 255).astype(np.uint8)
        sq = pad_to_square(masked, pad_color=bg_bgr)
        return cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    if rep_name == "rep4_rotated_masked_crop":
        hs, _ = seg_res.strip_bgr.shape[:2]
        ys_pos = np.where(seg_res.strip_mask_bin > 0)[0]
        if ys_pos.size > 0:
            yc_min = int(np.min(ys_pos))
            yc_max = int(np.max(ys_pos))
            margin = max(8, int(round(max(10.0, seg_res.mean_half_width_px * 2.2))))
            r1 = max(0, yc_min - margin)
            r2 = min(hs, yc_max + margin + 1)
        else:
            r1, r2 = max(0, hs // 4), min(hs, (3 * hs) // 4)
        if r2 <= r1:
            r1, r2 = 0, hs
        strip_crop = seg_res.strip_bgr[r1:r2, :].astype(np.float32)
        alpha = seg_res.strip_mask_soft[r1:r2, :, None]
        blur_strip = cv2.GaussianBlur(strip_crop, (15, 15), 5.0).astype(np.float32)
        bg = 0.15 * blur_strip + 0.85 * np.full_like(strip_crop, np.array(bg_bgr, dtype=np.float32))
        # Preserve pure shaft + faint immediate rim boundary so edge sharpness stays informative to ViT
        rim = (
            cv2.dilate(
                (seg_res.strip_mask_bin[r1:r2, :] > 0).astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
            ).astype(np.float32)[:, :, None]
            * 0.25
        )
        alpha_eff = np.clip(np.maximum(alpha, rim), 0.0, 1.0)
        masked_strip = np.clip(strip_crop * alpha_eff + bg * (1.0 - alpha_eff), 0, 255).astype(np.uint8)
        sq = pad_to_square(masked_strip, pad_color=bg_bgr)
        return cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    if rep_name == "rep5_adaptive_pad_crop":
        # Tightly crop strip height around predicted active mask +/- adaptive thickness margin
        strip = seg_res.strip_bgr
        hs, ws = strip.shape[:2]
        ys_pos = np.where(seg_res.strip_mask_bin > 0)[0]
        if ys_pos.size > 0:
            yc_min = int(np.min(ys_pos))
            yc_max = int(np.max(ys_pos))
            margin = max(6, int(round(max(8.0, seg_res.mean_half_width_px * 1.8))))
            r1 = max(0, yc_min - margin)
            r2 = min(hs, yc_max + margin + 1)
        else:
            r1, r2 = max(0, hs // 4), min(hs, (3 * hs) // 4)
        tight = strip[r1:r2, :]
        sq = pad_to_square(tight, pad_color=bg_bgr)
        return cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # rep6_minimal_corridor_crop: multi-strip 3-band longitudinal stacking into 224x224
    # Allows long 1:18 cane shafts to fill >45% of DINOv2 tokens cleanly!
    strip = seg_res.strip_bgr.astype(np.float32)
    alpha = np.clip(seg_res.strip_mask_soft[:, :, None] * 0.85 + 0.15, 0.0, 1.0)
    bg = np.full_like(strip, np.array(bg_bgr, dtype=np.float32))
    blend = np.clip(strip * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    hs, ws = blend.shape[:2]
    ys_pos = np.where(seg_res.strip_mask_bin > 0)[0]
    if ys_pos.size > 0:
        ymid = int(round(float(np.median(ys_pos))))
        half_band = max(12, int(round(max(12.0, seg_res.mean_half_width_px * 2.2))))
        r1 = max(0, ymid - half_band)
        r2 = min(hs, ymid + half_band + 1)
        corridor = blend[r1:r2, :]
    else:
        corridor = blend

    # Split corridor horizontally into 3 overlapping segments: [handle/top, mid, red-tip/bottom]
    c_w = corridor.shape[1]
    s1 = corridor[:, :max(16, int(0.42 * c_w))]
    s2 = corridor[:, max(0, int(0.29 * c_w)):min(c_w, max(20, int(0.71 * c_w)))]
    s3 = corridor[:, max(0, int(0.58 * c_w)):]
    band_h = target_size // 3
    p1 = cv2.resize(s1, (target_size, band_h), interpolation=cv2.INTER_LINEAR)
    p2 = cv2.resize(s2, (target_size, band_h), interpolation=cv2.INTER_LINEAR)
    p3 = cv2.resize(s3, (target_size, target_size - 2 * band_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.vstack([p1, p2, p3])
    return canvas


def canvas_to_dinov2_tensor(canvas_bgr: np.ndarray) -> np.ndarray:
    """Convert 224x224 BGR canvas to NCHW RGB ImageNet-normalized float32."""
    if canvas_bgr.shape[:2] != (224, 224):
        canvas_bgr = cv2.resize(canvas_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(normed, (2, 0, 1))[None, ...], dtype=np.float32)


class DINOv2EmbedderV2:
    """Frozen DINOv2-Small INT8 CPU ONNX wrapper returning L2-normalized 384D embeddings."""

    def __init__(self, model_path: Path, num_threads: int = 4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing DINOv2 model: {self.model_path}")
        if compute_sha256(self.model_path) != DINOV2_SHA256:
            raise ValueError(f"DINOv2 SHA-256 mismatch for {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def embed_canvas(self, canvas_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        tensor = canvas_to_dinov2_tensor(canvas_bgr)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        feat = raw[:, 0, :] if raw.ndim == 3 else raw
        emb = _l2_normalize(feat[0])
        ms = (time.perf_counter() - t0) * 1000.0
        return emb, round(ms, 2)

    def embed_raw_roi(self, roi_bgr: np.ndarray, size: int = 224) -> tuple[np.ndarray, float]:
        """Standard aspect-preserving ROI embedding compatible with VL-P1c R5 classes object_02..05."""
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("Empty ROI")
        work = roi_bgr
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        elif work.shape[2] == 4:
            work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
        border = np.concatenate([work[0, :], work[-1, :], work[:, 0], work[:, -1]], axis=0)
        med = tuple(int(c) for c in np.median(border, axis=0))
        sq = pad_to_square(work, pad_color=med)
        interp = cv2.INTER_AREA if min(sq.shape[:2]) > size else cv2.INTER_LINEAR
        resized = cv2.resize(sq, (size, size), interpolation=interp)
        return self.embed_canvas(resized)


def _build_whole_roi_proposal(img: np.ndarray) -> OrientedCaneProposal:
    """Construct a whole-ROI oriented proposal from foreground PCA or ROI diagonal for cropped shots."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch = lab[:, :, 0] * (100.0 / 255.0)
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_ch, b_ch)
    fg = ((l_ch >= 60.0) & (chroma <= 24.0)) | (a_ch >= 14.0)
    ys, xs = np.where(fg)

    p1 = p2 = None
    if len(xs) >= 16:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        mean_pt = pts.mean(axis=0)
        cov = np.cov((pts - mean_pt).T)
        vals, vecs = np.linalg.eigh(cov)
        if vals[-1] / max(1e-3, vals[0]) >= 3.0:
            u = vecs[:, -1]
            proj = (pts - mean_pt) @ u
            p1 = mean_pt + u * float(np.percentile(proj, 2))
            p2 = mean_pt + u * float(np.percentile(proj, 98))

    if p1 is None or p2 is None or float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) < 16.0:
        if w > 1.8 * h:
            p1 = np.array([0.05 * w, 0.5 * h], dtype=np.float32)
            p2 = np.array([0.95 * w, 0.5 * h], dtype=np.float32)
        elif h > 1.8 * w:
            p1 = np.array([0.5 * w, 0.05 * h], dtype=np.float32)
            p2 = np.array([0.5 * w, 0.95 * h], dtype=np.float32)
        else:
            p1 = np.array([0.06 * w, 0.06 * h], dtype=np.float32)
            p2 = np.array([0.94 * w, 0.94 * h], dtype=np.float32)

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    thick = max(8.0, min(22.0, 0.04 * float(min(h, w))))
    obb, corners = obb_from_endpoints(x1, y1, x2, y2, thick)
    cl_pts = [[round(x1 + f * (x2 - x1), 2), round(y1 + f * (y2 - y1), 2)] for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return OrientedCaneProposal(
        bbox_xywh=(0, 0, int(w), int(h)),
        obb_cxcywh_angle=obb,
        obb_corners_xy=corners,
        endpoints_xyxy=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        centerline_points_xy=cl_pts,
        confidence=1.0,
        bilateral_profile_score=1.0,
        white_purity_score=1.0,
        red_tip_score=0.0,
        hand_anchor_score=0.0,
        border_penalty=0.0,
        collinear_bridge_count=0,
        est_thickness_px=round(thick, 2),
        source="roi_principal_axis",
    )


class RepresentationEmbeddingBank:
    """Enrolls `train_pos/object_01` and negative prototypes in matching query representation."""

    def __init__(self, embedder: DINOv2EmbedderV2) -> None:
        self.embedder = embedder
        self.prototypes_by_rep: dict[str, np.ndarray] = {}
        self.shot_embeddings_by_rep: dict[str, np.ndarray] = {}
        self.neg_gallery_by_rep: dict[str, np.ndarray] = {}

    def fit_cane_enrollment_shots(
        self,
        train_cane_items: list[dict],
        localizer: OrientedCaneLocalizer,
        segmenter: CaneShaftSegmenter,
    ) -> dict[str, int]:
        """Extract matched representations on train_pos/object_01 (proposals + whole-ROI principal axis)."""
        counts = {}
        cached_inputs: list[tuple[np.ndarray, list[OrientedCaneProposal]]] = []
        for it in train_cane_items:
            img = load_image_unicode(it["path"])
            props, _ = localizer.detect_with_context(img, top_k=3)
            roi_prop = _build_whole_roi_proposal(img)
            cand_props = list(props[:2]) + [roi_prop]
            cached_inputs.append((img, cand_props))

        for rep in REPRESENTATION_NAMES:
            embs = []
            need_full = (rep == "rep3_masked_crop")
            for img, cand_props in cached_inputs:
                for p in cand_props:
                    seg = segmenter.segment_proposal(img, p, return_full_mask=need_full)
                    canvas = build_representation_canvas(img, p, seg, rep)
                    e, _ = self.embedder.embed_canvas(canvas)
                    embs.append(e)
            mat = np.vstack(embs).astype(np.float32)
            self.shot_embeddings_by_rep[rep] = mat
            self.prototypes_by_rep[rep] = _l2_normalize(np.mean(mat, axis=0))
            counts[rep] = int(mat.shape[0])
        return counts

    def score_embedding_against_cane(self, emb: np.ndarray, rep_name: str) -> tuple[float, float]:
        """Return (blended_pos_sim, max_cos_to_enrolled_shots)."""
        proto = self.prototypes_by_rep[rep_name]
        shots = self.shot_embeddings_by_rep[rep_name]
        sims = shots @ emb
        sim_proto = float(np.dot(emb, proto))
        top2_sorted = np.sort(sims)[-min(2, len(sims)):]
        top2_shot = float(np.mean(top2_sorted))
        sim_shot = float(top2_sorted[-1])
        blended = 0.35 * sim_proto + 0.65 * top2_shot
        return round(blended, 4), round(sim_shot, 4)
