# VL-P2 Closure & Post-Optimization Regression Audit (Gate 0)

**Auditor:** Google Antigravity (VL-P3 Gate 0 Independent Audit)  
**Target Protected Baseline:** Commit `4a10113` (`4a1011308ce8d63cee279354dfcc2ea68d9bb7cd`, branch `research/vl-p2-white-cane-segmentation`)  
**Machine-Readable Evidence:** [`manifests/vl_p2_post_opt_regression_audit.json`](manifests/vl_p2_post_opt_regression_audit.json)

---

## 1. Purpose of Gate 0

Before beginning VL-P3 model design, VL-P3 Gate 0 audited two concrete closure risks in VL-P2 (`4a10113`):

1. **Git Baseline & Worktree Integrity:** Verify all 8 protected historical commits, branches, ancestry chains, and clean worktree states.
2. **Post-Optimization Behavioral Regression:** Verify whether the late speed optimizations committed in `4a10113` (`localization_research.py` downscale to 360px + top-48/top-24 segment pruning, `combined_pipeline_v2.py` `props[:3]` truncation + `pos_sim >= 0.52` early exit) preserved the locked Phase B `C2_full_cane_verified` predictions across **all 180 images** of `final_holdout_v2`, and verify full-holdout latency against the 8-image `test_12` number (`153.09 ms`).

All audits ran **strictly read-only** against `VisionMate_VL_P2_AG` without modifying any protected commit or file.

---

## 2. Git Baseline Preservation Audit

All 8 protected commits, expected branch refs, and linear/ancestry chains passed:

| Milestone | Short SHA | Full SHA-1 | Branch | Ancestry to `HEAD` | Worktree Clean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| English Tables & TTS | `8ae6f85` | `8ae6f85b5ed4e38bf343ca623d0b02f2fbf99ec9` | `hotfix/english-tables-layout-and-tts` | `8ae6f85->95a75e0` | N/A |
| OD-1 Foundation | `95a75e0` | `95a75e0adeff322e19a6676cc6d1146e99afba7e` | `hotfix/od-1-foundation` | `95a75e0->6e3039f` | N/A |
| Static Preview | `6e3039f` | `6e3039f97f8065153dc96c4affb4f9371d3bcc7e` | `research/vl-p1-manual` | `6e3039f->65c2a27` & `988fdf1` | Manual worktree (`?? research/`) |
| Webcam FPS Optimization | `65c2a27` | `65c2a27a21fac65e5e3de78a4feba081dd9e192e` | `hotfix/webcam-preview-fps-optimization` | `6e3039f->65c2a27` | Yes |
| VL-P1 | `988fdf1` | `988fdf14a574863f6246017c5ccf8350eef5fdb8` | `research/vl-p1-antigravity` | `6e3039f->988fdf1` | Yes |
| VL-P1b | `dacd245` | `dacd24571ae2c1f6fe51db75e832c0bc5d622215` | `research/vl-p1b-reliability-proposal` | `988fdf1->dacd245` | Yes |
| VL-P1c | `9587ad4` | `9587ad491b62323f2196bd1749994e7b9d140deb` | `research/vl-p1c-calibration-multiproposal` | `dacd245->9587ad4` | Yes |
| VL-P2 | `4a10113` | `4a1011308ce8d63cee279354dfcc2ea68d9bb7cd` | `research/vl-p2-white-cane-segmentation` | `9587ad4->4a10113->HEAD` | Yes |

- `p2_out_of_scope_diff_vs_9587ad4`: empty (`""`)
- `p3_out_of_scope_diff_vs_4a10113`: empty (`""`)
- `git_tags`: empty (`[]`)

---

## 3. VL-P2 Post-Optimization Regression Results (All 180 `final_holdout_v2` Images)

Running the committed `4a10113` `C2_full_cane_verified` code unchanged over all 180 `final_holdout_v2` images reveals a **severe recall regression** compared with VL-P2's locked pre-optimization `benchmark_metrics.json`:

| Metric | VL-P2 Locked Phase B (`benchmark_metrics.json`) | Committed `4a10113` Measured (`N = 180`) | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | `145/180` (`80.6%`, 95% CI `[74.16, 85.67]`) | `133/180` (`73.9%`, 95% CI `[67.02, 79.76]`) | **-6.7 pts** |
| **Known Recall (`object_01`)** | `30/51` (`58.8%`, 95% CI `[45.17, 71.25]`) | **`10/51`** (`19.6%`, 95% CI `[11.02, 32.46]`) | **-39.2 pts** |
| **False Alarm Rate (FAR)** | `14/129` (`10.8%`, 95% CI `[6.58, 17.39]`) | `6/129` (`4.7%`, 95% CI `[2.15, 9.78]`) | -6.2 pts |
| **Unknown Specificity** | `115/129` (`89.1%`) | `123/129` (`95.3%`, 95% CI `[90.22, 97.85]`) | +6.2 pts |
| **Ambiguous Rate** | `112/180` (`62.22%`: `19` pos + `93` neg) | `114/180` (`63.33%`: `30` pos + `84` neg) | +1.11 pts |
| **Positive Status Agreement** | Reference | **`24/51` identical, `27/51` changed** | **Behavioral change** |
| **Latency `loc P50`** | `746.74 ms` | `201.49 ms` (`loc P95 = 316.09 ms`) | -545.25 ms |
| **Latency `seg P50`** | `125.00 ms` | `74.62 ms` | -50.38 ms |
| **Latency `emb P50`** | `195.93 ms` | `134.53 ms` | -61.40 ms |
| **Latency `ver P50`** | `29.44 ms` | `11.31 ms` | -18.13 ms |
| **Total E2E `P50 / P95`** | `1107.80 / 2040.35 ms` | **`420.95 / 645.89 ms`** | **`153.09 ms` was dev-subset only** |

### Per-Category Breakdown in Post-Optimization `4a10113`
- **Per-Scenario Recall (`object_01`):**
  - `handheld`: `10/47` (`21.3%`, 95% CI `[11.99, 34.90]`)
  - `occluded`: `6/28` (`21.4%`, 95% CI `[10.21, 39.54]`)
  - `red_tip`: `0/3` (`0.0%`, 95% CI `[0.00, 56.15]`)
  - `folded`: `0/1` (`0.0%`, 95% CI `[0.00, 79.35]`)
- **Per-Negative-Group FAR (`unknown`):**
  - `walking_stick_confuser`: **`3/19`** (`15.8%`, 95% CI `[5.52, 37.57]`) — highest-risk failure group (`v2_neg_walking__013` score `0.9403`, `v2_neg_walking__020` score `0.6303`, `v2_neg_walking__031` score `0.9753`)
  - `metal_rod_pipe_confuser`: `1/19` (`5.3%`, 95% CI `[0.94, 24.64]`, `v2_neg_metal_ro_022` `0.9409`)
  - `general_distractor`: `1/24` (`4.2%`, 95% CI `[0.74, 20.24]`, `v2_neg_general__007` `0.7932`)
  - `white_line_pole_confuser`: `1/26` (`3.9%`, 95% CI `[0.68, 18.89]`, `v2_neg_white_li_026` `0.7734`)
  - `trekking_pole_confuser`: `0/20` (`0.0%`)
  - `umbrella_confuser`: `0/21` (`0.0%`)

---

## 4. Code-Level Root Cause Analysis

Inspecting `VisionMate_VL_P2_AG/research/vl_p2/src/localization_research.py` and `combined_pipeline_v2.py` confirms why `27/51` positive predictions changed and latency averaged `420.95 ms` rather than `153.09 ms`:

1. **Resolution & Segment Truncation (`localization_research.py`):**
   - `scale = min(1.0, 360.0 / max(h, w, 1))` shrunk working resolution to `360px` long-side, attenuating thin (2–4px wide) distant or oblique canes.
   - `_extract_raw_segments` capped candidate segments to top `48` (`segments = [segments[i] for i in top_idx]`), `_bridge_collinear_segments` sliced `seg_meta[:48]`, and `detect_with_context` evaluated only `bridged[:24]`.
2. **Candidate Loop Truncation & Defective Early Exit (`combined_pipeline_v2.py`):**
   - `predict()` sliced `props[:min(3, self.cfg.top_k_proposals)]`, silently dropping candidate #4 when `selected_top_k == 4`.
   - Line 201 executed `if n_eval >= 2 and best_any is not None and best_any[1]["pos_sim"] >= 0.52: break` **even when `dec["status"]` was `AMBIGUOUS` or `UNKNOWN`**, immediately aborting search before inspecting candidate #3. Notice several regressed positives (`v2_ovc_068` score `0.9036`, `v2_ov3_047` `0.8757`, `v2_ovc_003` `0.8676`, `v2_ovc_066` `0.8429`, `v2_ovc_062` `0.7609`) had high verifier probabilities yet returned `AMBIGUOUS` because one of the un-selected or 360px-degraded crops missed `tau_cane_sim` / `min_white_purity` or triggered the early `break`.
3. **Dev-Only Latency Claim:** The `153.09 ms` figure in VL-P2 `test_12` came from 8 selected dev images (`test_performance_speed_and_memory`) rather than full `final_holdout_v2` (`420.95 ms` P50 / `645.89 ms` P95).

**VL-P3 Governance Decision:** VL-P3 must never cherry-pick latency subsets or ship late "optimizations" after Phase B freeze. Latency, memory, and accuracy must all be measured on the complete blind holdout from the exact frozen configuration.
