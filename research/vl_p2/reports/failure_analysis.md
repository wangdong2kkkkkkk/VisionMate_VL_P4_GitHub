# Failure Analysis & Root-Cause Diagnosis — VL-P2

All findings below are derived from `research/vl_p2/manifests/frozen_config_v2.json` and `research/vl_p2/reports/benchmark_metrics.json`.

## 1. Error Inventory on Blind Holdout (`C2_full_cane_verified`, `N = 180`)

### False Negatives (`21 / 51` missed positives)
- **`FN_UNKNOWN` (`2`):** `v2_ecane_023` (`pos_sim = 0.6681`), `v2_ov3_050` (`pos_sim = 0.1768`).
- **`FN_AMBIGUOUS` (`19`):** `v2_ecane_015` (`0.0551`), `v2_fcane_002` (`0.6728`), `v2_fcane_003` (`0.8046`), `v2_pos_cane_003` (`0.6471`), `v2_pos_cane_066` (`0.8282`), `v2_scane_016` (`0.8808`), `v2_scane_018` (`0.0115`), `v2_scane_021` (`0.0097`), `v2_dcane_010` (`0.8771`), `v2_dcane_011` (`0.4869`), `v2_glc_008` (`0.4400`), `v2_ovc_002` (`0.8903`), `v2_ovc_006` (`0.7916`), `v2_ovc_021` (`0.5602`), `v2_ovc_067` (`0.7827`), `v2_ov3_035` (`0.5906`), `v2_ov3_039` (`0.8750`), `v2_ov3_052` (`0.9230`), `v2_ov3_085` (`0.6901`).
- Notable pattern: most false negatives (`19/21`) are safely abstained as `AMBIGUOUS` rather than rejected outright, and many (e.g., `v2_ov3_052` at `0.9230`, `v2_ovc_002` at `0.8903`, `v2_scane_016` at `0.8808`) exhibit high `pos_sim` but fall short of the calibrated `prob_known >= 0.62` or `delta_neg_min >= -0.02` margin against hard confusers.

### False Positives (`14 / 129` negatives, `FAR = 10.8%`)
- Category distribution (`per_neg_group_far`):
  - `walking_stick_confuser`: `3/19` (`15.8%`)
  - `metal_rod_pipe_confuser`: `3/19` (`15.8%`)
  - `trekking_pole_confuser`: `3/20` (`15.0%`)
  - `umbrella_confuser`: `3/21` (`14.3%`)
  - `general_distractor`: `1/24` (`4.2%`)
  - `white_line_pole_confuser`: `1/26` (`3.9%`)
- Notice `C2` kept `white_line_pole_confuser` FAR at `1/26 (3.9%)` (same as `A0` and lower than `B1`'s `3/26 = 11.5%`), proving white-purity & bilateral shaft verification filters flat painted road lines reasonably well, whereas **handheld slender objects** (`walking_stick`, `trekking_pole`, `umbrella`, `metal_rod_pipe`) account for `12/14` false alarms.

## 2. Four Root Causes Identified

1. **Classical LSD/Hough Localization Ceiling in Cluttered Scenes:**
   - On blind `final_holdout_v2`, `A1_oriented_cane_k4` achieves only `2/51 (3.9%)` `obb_recall_0_3` (`handheld: 1/47 = 2.1%`, `occluded: 1/28 = 3.6%`), pulling `mean_post_mask_bg_contamination` up to `0.9315`. Without an accurate oriented proposal on unseen backgrounds, `CaneShaftSegmenter` refines an inaccurate corridor.
2. **DINOv2-Small Representation Collapse Under Tight Masking & Rotation:**
   - Removing context via tight rotated masked strips (`rep4_rotated_masked_crop`) causes both `object_01` and linear confusers to converge to generic diagonal stripe embeddings (`holdout pos_sim = 0.6635` vs `neg_sim = 0.6437`, `gap = +0.0157`, `recall@5% = 5.88%`). Soft attenuated masking (`rep3_masked_crop`) mitigates this (`gap = +0.0809`) while remaining sensitive to noisy proposals.
3. **High Verifier Reliance on Pedestrian Anchor (`hand_person_anchor`) vs Fine Appearance Cues:**
   - OOF logistic regression weights (`manifests/frozen_config_v2.json`) are dominated by `hand_person_anchor (+1.6155)` and `dinov2_confuser_margin (+1.2339)`, compared to `red_tip_evidence (+0.4731)`, `white_purity_core (+0.3473)`, and modest negative weights (`canopy_flare_penalty -0.1395`, `wood_brown_ratio -0.1188`, `metallic_dark_ratio -0.0872`). When a pedestrian holds a walking stick, umbrella, or trekking pole, `hand_person_anchor` boosts its score toward `prob_known >= 0.62`.
4. **Operating-Point Tradeoff Between Recall & FAR:**
   - Moving from `A0` / `C1` (`41.2–43.1%` recall) to `C2` (`58.8%` recall) yields `+15.69 pp` positive recall (`McNemar 11 new wins vs 3 ref wins, p = 0.057373`) while trading `-8.53 pp` negative specificity (`p = 0.003418`, `FAR: 2.3% -> 10.8%`). On the historical 5-class suite (`N=300`), however, `object_01` recall jumps `+55.0 pp` (`10.0% -> 65.0%`) with only a `+2.0 pp` overall FAR shift (`5.5% -> 7.5%`).
