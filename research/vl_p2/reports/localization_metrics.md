# Oriented White-Cane Localization Metrics — VL-P2

## 1. Proposer Design (`src/localization_research.py`)

`OrientedCaneLocalizer` pairs YOLOX-Nano pedestrian boxes (`class_id == 0`) with orientation-aware classical structural extraction:

- Multi-scale morphological feature maps (`_compute_cane_feature_maps`).
- OpenCV LSD + Probabilistic Hough segment extraction (`_extract_raw_segments`).
- Collinear segment gap-bridging (`_bridge_collinear_segments`).
- Ridge-center snapping, gap<=4 contiguous run validation, outward occlusion stepping, and cross-sectional contrast verification (`_evaluate_and_refine_candidate`).
- Person/hand proximity anchoring and normalized OBB thickness estimation (`detect_with_context`).

## 2. Quantitative Localization Results (`dev_cal_v2` vs `final_holdout_v2`)

| Split | Method / Top-K | `n` | `axis_recall_0_3` | `axis_recall_0_5` | `obb_recall_0_3` | `obb_recall_0_5` | `endpoint_inclusion` | `mean_axis_iou` | `mean_obb_iou` | `mean_angle_err_deg` | `median_angle_err_deg` | `mean_axis_bg_contam` | `mean_obb_bg_contam` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dev_cal_v2` | `A1_oriented_cane_k1` | 65 | — | 13/65 (20.0%) | 8/65 (12.3%) | 3/65 (4.6%) | 18/65 (27.7%) | 0.2455 | 0.1063 | 21.38° | 9.16° | 0.9689 | 0.8764 |
| `dev_cal_v2` | **`A1_oriented_cane_k4`** | 65 | — | **17/65 (26.1%)** | **15/65 (23.1%)** | 4/65 (6.2%) | 23/65 (35.4%) | 0.3047 | 0.1650 | 17.09° | 6.71° | 0.9341 | 0.8028 |
| `dev_cal_v2` | `A1_oriented_cane_k5` | 65 | — | 18/65 (27.7%) | 16/65 (24.6%) | 5/65 (7.7%) | 25/65 (38.5%) | — | — | — | — | — | — |
| `final_holdout_v2` | `A0_v1c_axis_proposer_k4` | 51 | 5/51 (9.8%) | 2/51 (3.9%) | — | — | — | 0.1116 | — | — | — | 0.9928 | — |
| `final_holdout_v2` | `A1_oriented_cane_k1` | 51 | — | 3/51 (5.9%) | 1/51 (2.0%) | 0/51 (0.0%) | 6/51 (11.8%) | 0.1275 | 0.0308 | 23.92° | 17.01° | 0.9877 | 0.9736 |
| `final_holdout_v2` | `A1_oriented_cane_k3` | 51 | — | 5/51 (9.8%) | 1/51 (2.0%) | — | 5/51 (9.8%) | 0.1875 | 0.0503 | — | 15.29° | — | — |
| `final_holdout_v2` | **`A1_oriented_cane_k4`** | 51 | — | **5/51 (9.8%)** | **2/51 (3.9%)** | 0/51 (0.0%) | 5/51 (9.8%) | 0.1864 | 0.0556 | 24.58° | 15.29° | 0.9778 | 0.9551 |
| `final_holdout_v2` | `A1_oriented_cane_k5` | 51 | — | 5/51 (9.8%) | 2/51 (3.9%) | — | — | 0.1918 | 0.0578 | — | — | — | — |

### Stratified Subset Recall (`A1_oriented_cane_k4`)

- `dev_cal_v2` (`k=4`): `occluded_obb_recall_0_3 = 5/18 (27.8%)`, `handheld_obb_recall_0_3 = 3/6 (50.0%)`, `mean_endpoint_err_norm = 0.4738`.
- `final_holdout_v2` (`k=4`): `occluded_obb_recall_0_3 = 1/28 (3.6%)`, `handheld_obb_recall_0_3 = 1/47 (2.1%)`, `mean_endpoint_err_norm = 0.7232`.
- Top-K selection (`manifests/frozen_config_v2.json`): `selected_top_k = 4` (chosen as the precision/latency sweet spot on dev: `95% CI` for `axis_recall_0_5` is `[17.02%, 37.95%]` on dev and `[4.26%, 20.98%]` on blind holdout).

## 3. Interpretation

1. **Improvement over VL-P1c axis proposer (`A0`):** On the blind holdout (`n=51`), `A1_oriented_cane_k4` raises `axis_recall_0_5` from `2/51 (3.9%)` to `5/51 (9.8%)` and `mean_axis_iou` from `0.1116` to `0.1864`, while reducing axis background contamination (`0.9928 -> 0.9778`).
2. **Severe dev-to-holdout domain drop:** Moving from curated `dev_cal_v2` scenes to complex real-world `final_holdout_v2` street/pedestrian images reduces `obb_recall_0_3` from `23.1%` to `3.9%` and doubles median angle error (`6.71° -> 15.29°`). Training-free LSD/Hough edge grouping frequently locks onto brighter clothes, pavement markings, railings, or limb edges when canes are thin, partially occluded, or low-contrast.
