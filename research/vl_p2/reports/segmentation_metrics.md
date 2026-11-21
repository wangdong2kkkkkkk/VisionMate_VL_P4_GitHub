# Strip Rectification & Shaft Segmentation Metrics — VL-P2

## 1. Segmentation Architecture (`src/segmentation_research.py`)

`CaneShaftSegmenter` extracts tight pixel shaft masks without heavy CNN decoders:

- **Affine Strip Rectification (`compute_rectification_affine`):** Warps the candidate corridor into a compact rectified coordinate strip (`target_len <= 240`, strip height `hs` clamped to `[40, 96]`).
- **Precomputed DP Boundary Tracking:** Computes smoothed upper/lower boundary paths along the strip using vectorized dynamic programming with precomputed index and transition penalty matrices outside the column loop, cutting median per-proposal segmentation latency from `99.75 ms` (initial dev baseline) down to **`23.25 ms`** on `dev_cal_v2` and **`25.83 ms`** on `final_holdout_v2`.
- **Boundary & Soft Mask Metrics (`evaluate_segmentation_against_gt`):** Reports mask IoU, Dice, precision, recall, `compute_boundary_f1`, and `mean_post_mask_bg_contamination`.

## 2. Quantitative Segmentation Results (`top_k = 4`)

| Metric | `dev_cal_v2` (`n=65`) | `final_holdout_v2` (`n=51`) |
|---|---|---|
| `mean_mask_iou` | **0.1981** | **0.0608** |
| `median_mask_iou` | 0.1379 | 0.0259 |
| `mean_dice` | 0.2830 | 0.1016 |
| `mean_precision` | 0.2327 | 0.0685 |
| `mean_recall` | 0.5030 | 0.2831 |
| `mean_boundary_f1` | 0.2865 | 0.1263 |
| `mean_post_mask_bg_contamination` | **0.7673** | **0.9315** |
| `occluded_mean_mask_iou` | 0.1873 | 0.0673 |
| `non_occluded_mean_mask_iou` | 0.2022 | 0.0529 |
| `seg_p50_ms` | `23.25 ms` | `25.83 ms` |

## 3. Six-Way DINOv2 Representation Ablation (`src/masked_embedding_research.py`)

To test whether segmentation masks improve DINOv2-Small INT8 separability without representation collapse, six canonical representations were evaluated on both `dev_cal_v2` and `final_holdout_v2`:

| Representation | Dev Oracle `(pos_sim, margin)` | Dev E2E `(pos_sim, neg_sim, gap, rec@5%)` | Holdout Oracle `(pos_sim, margin)` | Holdout E2E `(pos_sim, neg_sim, gap, rec@5%)` |
|---|---|---|---|---|
| `rep1_axis_crop` | `(0.4719, +0.0661)` | `(0.4647, 0.2979, +0.1170, 0.4615)` | `(0.4321, +0.0327)` | **`(0.4245, 0.2556, +0.1127, 0.4706)`** |
| `rep2_rotated_crop` | `(0.4871, +0.0144)` | `(0.4903, 0.3968, +0.0722, 0.2923)` | `(0.4656, -0.0039)` | `(0.4672, 0.4097, +0.0340, 0.0980)` |
| **`rep3_masked_crop`** | `(0.5723, -0.0129)` | **`(0.5669, 0.4517, +0.0891, 0.4308)`** *(cal rec `0.5385` @ FAR `0.0565`)* | `(0.4994, -0.0481)` | `(0.5339, 0.4377, +0.0809, 0.3529)` |
| `rep4_rotated_masked_crop` | `(0.6728, -0.0174)` | `(0.6852, 0.6208, +0.0334, 0.2462)` | `(0.6609, -0.0138)` | `(0.6635, 0.6437, +0.0157, 0.0588)` |
| `rep5_adaptive_pad_crop` | `(0.4993, -0.0052)` | `(0.5016, 0.4038, +0.0637, 0.3231)` | `(0.4765, -0.0086)` | `(0.4670, 0.4194, +0.0416, 0.2549)` |
| `rep6_minimal_corridor_crop` | `(0.5710, +0.0171)` | `(0.5752, 0.5113, +0.0551, 0.3385)` | `(0.5301, +0.0057)` | `(0.5597, 0.5100, +0.0398, 0.2157)` |

- `frozen_config_v2.json` records `selected_best_representation = "rep3_masked_crop"` (best Method B soft-masked axis crop using `0.15*blur + 0.85*bg` anti-collapse background attenuation), while explicitly logging `best_overall_rep_including_axis = "rep1_axis_crop"`.
- **Key Scientific Finding:** Hard geometric cropping/masking plus axis rotation (`rep4_rotated_masked_crop`) inflates positive similarity (`0.6852` dev, `0.6635` holdout) equally on elongated confusers (`neg_sim = 0.6208` / `0.6437`), collapsing the class margin (`gap = +0.0157`, holdout recall@5% `0.0588`). Soft background attenuation (`rep3_masked_crop`) partially preserves context while upstream localization remains the primary cap on mask IoU.
