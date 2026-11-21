# VISIONMATE GLASSES — VL-P2 FINAL RESEARCH REPORT
## White Cane Localization & Segmentation Research (`object_01`)

**Status:** Independent Research Pilot Complete (NOT integrated into production `app/`)  
**Worktree / Branch:** `VisionMate_VL_P2_AG` on `research/vl-p2-white-cane-segmentation` (branched from `9587ad4`)  
**Models:** DINOv2-Small INT8 (`3afdc8bc...`) + YOLOX-Nano (`c789161e...`) via ONNX Runtime CPU (Apache-2.0, 100% offline)

---

## 1. Executive Summary & Scientific Verdict

VL-P2 investigated why the prior VL-P1c axis-aligned proposal + global DINOv2 embedding pipeline failed on white canes (`object_01`), achieving only **2/20 (10.0%)** recall on `vl_p1c/final_holdout_full` (`N = 300`) due to high background contamination (`0.736`) inside axis bounding boxes and thin-object dilution against pedestrians and pavement.

Working strictly under the Phase A / Phase B blind protocol (`final_holdout_touched_during_calibration: false`), VL-P2 built a **189-image** development split (`dev_cal_v2`), a **180-image** blind holdout split (`final_holdout_v2`: `51` `object_01`, `129` hard linear/handled confusers), an **18-image** ambiguous abstention set (`ambiguous_cane_v2`), and **116** multi-geometry `object_01` annotations (`bbox_xywh`, rotated OBB, centerline/endpoints, RLE mask, occlusion tags).

**Headline Results:**
- **Historical 5-class benchmark (`vl_p1c/final_holdout_full`, `N = 300`):** Replacing only the `object_01` head with `VL-P2 C2_full_cane_verified` (keeping `VL-P1c R5` unchanged for `object_02`–`object_05`) lifted **`object_01` recall from `2/20 (10.0%)` to `13/20 (65.0%)`** (`+55.0 pp`), raising known recall from `62/100 (62.0%)` to `73/100 (73.0%)` and overall accuracy from `251/300 (83.7%)` to **`258/300 (86.0%)`**, with **zero regression** on `object_02` (`80.0%`), `object_03` (`70.0%`), `object_04` (`65.0%`), and `object_05` (`85.0%`).
- **Blind white-cane holdout (`final_holdout_v2`, `N = 180`):** `C2_full_cane_verified` (`rep3_masked_crop` + 12D out-of-fold verifier) lifted `object_01` recall from **`22/51 (43.1%)`** (`A0_axis_baseline`) to **`30/51 (58.8%)`** (`+15.69 pp`, handheld `59.6%`, occluded `46.4%`, McNemar positive `p = 0.057373`, `11` new wins vs `3` ref wins), while trading higher false alarm rate (`14/129 = 10.8%` vs `A0`'s `3/129 = 2.3%`, McNemar negative `p = 0.003418`), primarily on pedestrian-held walking sticks, trekking poles, umbrellas, and metal rods.
- **Verdict:** **Research Pilot only — NOT ready for production integration.** Classical training-free LSD/Hough oriented localization dropped sharply on cluttered blind scenes (`obb_recall_0_3: 23.1%` on dev -> `3.9%` on blind holdout), leaving too much residual background inside candidate strips (`mean_post_mask_bg_contamination = 0.9315`) and forcing reliance on person-proximity anchoring that conflates handheld confusers with white canes.

---

## 2. Reproducing Protected Baselines (`A0` & `C_v1c_cal_only`)

| Baseline | Dataset (`N`) | Overall Acc | `object_01` Recall | Negative FAR | Bg Contamination |
|---|---|---|---|---|---|
| `C_v1c_cal_only` | `vl_p1c/final_holdout_full` (`300`) | `251/300` (`83.7%`) `[79.06, 87.42]` | `2/20` (`10.0%`) | `11/200` (`5.5%`) | `0.736` (`obj01_mean_min_bg_contam`) |
| `A0_axis_baseline` | `final_holdout_v2` (`180`) | `148/180` (`82.2%`) `[75.98, 87.12]` | `22/51` (`43.1%`) `[30.50, 56.73]` | `3/129` (`2.3%`) `[0.79, 6.61]` | `0.9928` (`mean_axis_bg_contamination`) |

*(See [MODEL_PROVENANCE.md](local historical file reference omitted), [DATASET_PROTOCOL.md](local historical file reference omitted), [ANNOTATION_PROTOCOL.md](local historical file reference omitted), and [git_baseline_preservation.md](local historical file reference omitted).)*

---

## 3. Methods A, B, and C Summary

- **Method A (`OrientedCaneLocalizer`):** YOLOX-Nano person detection (`class_id == 0`) + multi-scale morphological feature maps + OpenCV LSD & Probabilistic Hough + collinear gap-bridging (`_bridge_collinear_segments`) + ridge-center snapping and outward occlusion stepping (`_evaluate_and_refine_candidate`, `selected_top_k = 4`).
- **Method B (`CaneShaftSegmenter` + 6-Way Representations):** Affine strip rectification + vectorized DP shaft boundary tracking (`seg_p50_ms = 23.25 ms` dev, `25.83 ms` holdout) followed by six canonical DINOv2-Small INT8 input canvases (`rep1_axis_crop` through `rep6_minimal_corridor_crop`). Phase A froze `selected_best_representation = "rep3_masked_crop"` (`0.15*blur + 0.85*bg` anti-collapse soft masking) while logging `best_overall_rep_including_axis = "rep1_axis_crop"`.
- **Method C (`CaneVsConfuserVerifier`):** 5-fold group-aware OOF logistic regression (`oof_n = 188`, `oof_pos_recall = 0.5846`, `oof_neg_far = 0.0244`) combining DINOv2 top-2/top-3 cosine margins (`selected_tau_cane_sim = 0.34`, `selected_delta_neg_min = -0.02`, `selected_min_white_purity = 0.14`, `selected_prob_known = 0.62`) with physical shaft cues (`hand_person_anchor +1.6155`, `dinov2_confuser_margin +1.2339`, `red_tip_evidence +0.4731`, `dinov2_pos_sim +0.4348`, `white_purity_core +0.3473`, `canopy_flare_penalty -0.1395`, `wood_brown_ratio -0.1188`, `metallic_dark_ratio -0.0872`).

---

## 4. Phase B Blind Results (`final_holdout_v2`, `N = 180` & Historical `N = 300`)

Detailed tables appear in [localization_metrics.md](local historical file reference omitted), [segmentation_metrics.md](local historical file reference omitted), [confusion_matrices.md](local historical file reference omitted), [failure_analysis.md](local historical file reference omitted), and [performance_report.md](local historical file reference omitted).

---

## 5. Mandatory Research Questions (Section XII.2)

1. **Why did `VL-P1c` fail badly on white canes (`2/20 = 10.0%`), and how much background contamination actually caused it?**  
   Axis-aligned bounding boxes enclosing diagonal white canes contain overwhelming background pavement, clothing, and pedestrian context: on `vl_p1c/final_holdout_full`, baseline `C_v1c_cal_only` recorded `obj01_mean_min_bg_contam = 0.736`, and on `final_holdout_v2`, axis proposer `A0_v1c_axis_proposer_k4` averaged **`0.9928`** background contamination (`99.28%` non-cane pixels). DINOv2-Small INT8 global embeddings averaged across those boxes were dominated by background semantics rather than the narrow cane shaft, yielding only `2/20 (10.0%)` recall on `vl_p1c/final_holdout_full` and `22/51 (43.1%)` on `final_holdout_v2`.

2. **How effectively does Method A (`OrientedCaneLocalizer`) locate diagonal, handheld, and partially occluded canes via oriented geometry compared to `A0`?**  
   On `dev_cal_v2` (`n = 65`), `A1_oriented_cane_k4` reached `axis_recall_0_5 = 17/65 (26.1%)`, `obb_recall_0_3 = 15/65 (23.1%)` (`handheld: 3/6 = 50.0%`, `occluded: 5/18 = 27.8%`), `endpoint_inclusion = 23/65 (35.4%)`, and median angle error `6.71°`. On blind `final_holdout_v2` (`n = 51`), `A1_oriented_cane_k4` outperformed `A0_v1c_axis_proposer_k4` on `axis_recall_0_5` (`5/51 = 9.8%` vs `2/51 = 3.9%`), `mean_axis_iou` (`0.1864` vs `0.1116`), and background contamination (`0.9778` axis / `0.9551` OBB vs `0.9928`), while `obb_recall_0_3` fell sharply to `2/51 (3.9%)` (`median_angle_err_deg = 15.29°`), revealing the domain-generalization limit of classical LSD/Hough proposals on cluttered blind streets.

3. **How accurate and fast is Method B (`CaneShaftSegmenter`) in isolating the cane shaft and lowering background contamination?**  
   Vectorized precomputed DP tracking runs at `seg_p50_ms = 23.25 ms` (dev) and `25.83 ms` (blind holdout) per proposal (after optimization from `99.75 ms`). On `dev_cal_v2` (`n = 65`), `CaneShaftSegmenter` achieved `mean_mask_iou = 0.1981` (`mean_dice = 0.2830`, `mean_boundary_f1 = 0.2865`) and cut background contamination from `0.9341` (axis) down to **`0.7673`**. On blind `final_holdout_v2`, upstream missed localizations (`obb_recall_0_3 = 3.9%`) lowered end-to-end mask quality to `mean_mask_iou = 0.0608` (`mean_post_mask_bg_contamination = 0.9315`).

4. **Does tight masking and/or axis rotation collapse DINOv2-Small representation diversity, and which representation works best?**  
   Yes — **representation collapse is directly observed when rotation and hard background removal strip all surrounding cues**: in `dev_cal_v2` and `final_holdout_v2`, `rep4_rotated_masked_crop` pushes raw positive cosine similarity to `0.6852` / `0.6635` while simultaneously pushing negative confuser similarity to `0.6208` / `0.6437`, shrinking the class separation gap to `+0.0334` (dev) and **`+0.0157`** (holdout), with `recall@5% FAR = 5.88%` (`B2` end-to-end recall drops to `6/51 = 11.8%`). On `final_holdout_v2` ranking by embedding gap alone, unmasked `rep1_axis_crop` retained the largest margin (`+0.1127`, `recall@5% = 47.06%`), while among masked variants `rep3_masked_crop` (`0.15*blur + 0.85*bg` anti-collapse blend) performed best (`gap = +0.0809` holdout, `+0.0891` dev; `recall@5% = 43.08%` dev / `35.29%` holdout).

5. **Does end-to-end recognition (`B1` / `C1` / `C2`) outperform `A0_axis_baseline` on the blind holdout (`N = 180`), and with what statistical significance?**  
   Using `rep3_masked_crop` alone without the structural verifier (`B1` & `C1`) gives lower accuracy (`140/180 = 77.8%`, `recall = 21/51 = 41.2%`, `FAR = 10/129 = 7.8%`) than `A0` (`148/180 = 82.2%`, `recall = 22/51 = 43.1%`, `FAR = 3/129 = 2.3%`, McNemar all `p = 0.115318`). Adding the 12D OOF structural verifier (`C2_full_cane_verified`) **raises `object_01` recall by `+15.69 pp`** (`30/51 = 58.8%` vs `22/51 = 43.1%`; handheld `59.6%` vs `40.4%`, occluded `46.4%` vs `28.6%`; McNemar positive `p = 0.057373`, **11 new wins vs 3 reference wins**), at an overall accuracy of `145/180 = 80.6%` (McNemar all `p = 0.701108`, `-1.67 pp`) because negative FAR rises to `14/129 = 10.8%` (McNemar negative `p = 0.003418`, `-8.53 pp`).

6. **How well does `C2_full_cane_verified` separate real white canes from visually similar confusers (`walking_stick`, `trekking_pole`, `umbrella`, `metal_rod_pipe`, `white_line_pole`)?**  
   Physical and bilateral verification keeps `white_line_pole_confuser` FAR at `1/26 (3.9%)` (matching `A0` and outperforming `B1/C1`'s `3/26 = 11.5%`) and `general_distractor` FAR at `1/24 (4.2%)`. However, **handheld cylindrical confusers** account for `12/14` false positives (`walking_stick: 3/19 = 15.8%`, `metal_rod_pipe: 3/19 = 15.8%`, `trekking_pole: 3/20 = 15.0%`, `umbrella: 3/21 = 14.3%`) because `hand_person_anchor` carries the largest positive logistic weight (`+1.6155`).

7. **How safely does the 3-state gate (`KNOWN` / `AMBIGUOUS` / `UNKNOWN`) behave on `ambiguous_cane_v2` (`n = 18`) and hard holdout negatives?**  
   On `ambiguous_cane_v2`, `C2_full_cane_verified` abstained safely on **`16/18 (88.9%)`** images (`11` `AMBIGUOUS`, `5` `UNKNOWN`, `2` forced `KNOWN`), matching `A0`, `B1`, and `C1`, while `A1` and `B2` abstained on `18/18 (100.0%)`. On `final_holdout_v2`, `19` of the `21` missed `object_01` samples under `C2` were routed to `AMBIGUOUS` rather than `UNKNOWN`, and `93` of `129` holdout negatives were routed to `AMBIGUOUS` (`22` `UNKNOWN`, `14` `FP`).

8. **Does VL-P2 integrate cleanly with `VL-P1c R5` on the historical 5-class benchmark (`vl_p1c/final_holdout_full`, `N = 300`) without breaking `object_02`–`object_05`?**  
   Yes: integrating `VL-P2 C2` for `object_01` with `VL-P1c R5` for `object_02`–`object_05` lifts overall 5-class accuracy from `251/300 (83.7%)` to **`258/300 (86.0%)`**, known recall from `62/100 (62.0%)` to **`73/100 (73.0%)`**, and `object_01` recall from `2/20 (10.0%)` to **`13/20 (65.0%)`** (`+55.0 pp`), with exact `0.0 pp` regression on `object_02` (`16/20 = 80.0%`), `object_03` (`14/20 = 70.0%`), `object_04` (`13/20 = 65.0%`), and `object_05` (`17/20 = 85.0%`), while negative FAR moves slightly from `11/200 (5.5%)` to `15/200 (7.5%)`.

9. **Can the CPU-only Windows pipeline meet real-time research latency (`<= 300 ms`) and RAM (`<= 1.5 GB`) targets?**  
   Yes after hot-path optimization: while the full-resolution unoptimized Phase B batch runner averaged `1107.80 ms` E2E median (`peak RSS = 222.16 MB`), optimizing segment pre-filtering (`cane_energy * sqrt(L)` top-48), `360 px` localization work scale, local crop-space affine warping inside `rep3_masked_crop`, and early acceptance (`<= 3` proposal evaluations) brought `test_12_runtime_and_memory_budget` (8 `dev_cal_v2` images) down to **`153.09 ms` median** (`136.6–161.0 ms`), roughly 2x faster than the `300.0 ms` target within `222–250 MB` RAM.

10. **Is VL-P2 ready for production merge into `VisionMateGlasses`, and what must happen next?**  
    **No — remain Research Pilot.** Blind holdout `object_01` recall (`58.8%`) and `10.8%` FAR on handheld slender distractors remain unsuitable for blind-user safety warnings in production, and classical LSD/Hough localizers (`3.9%` blind OBB@0.3) cannot reliably isolate thin shafts on unconstrained street backgrounds. The next research step should replace classical LSD/Hough with a **lightweight supervised CPU thin-line / keypoint (handle-to-tip) cane detector + trained shaft/hand-grip segmenter**, keeping VL-P2's multi-geometry annotation schema, `ambiguous_cane_v2` safety audit, and 3-state verification architecture.
