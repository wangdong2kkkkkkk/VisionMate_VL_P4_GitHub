# Dataset Protocol & Zero-Leakage Audit — VL-P2

## 1. Split Topology & Image Quotas

All image splits, hashes, group identifiers, and provenance metadata are recorded in `research/vl_p2/manifests/dataset_manifest.json` (`sha256: 211ad32191fd8d64df139294200caae3fdb80528e513a6d5f2c4f73ab29349af`, `data_quota_status: "READY"`).

| Split | Purpose | Total Images | Positives (`object_01`) | Negatives / Non-Target | Calibration / Blind Rule |
|---|---|---|---|---|---|
| `train_pos` | Positive enrollment support gallery | 25 | 25 | 0 | Historical VL-P1 positive gallery |
| `hard_neg_gallery` | Elongated & handled hard-negative gallery | 50 | 0 | 50 | Reference negative prototypes for margin & verifier |
| `dev_cal_v2` | Development, representation ablation & 5-fold OOF verifier fit | 189 | 65 | 124 | Calibration & feature/threshold selection only |
| `final_holdout_v2` | Blind white-cane evaluation benchmark | 180 | 51 | 129 | Locked until Phase B (`final_holdout_touched_during_calibration: false`) |
| `ambiguous_cane_v2` | Boundary / low-evidence abstention safety suite | 18 | 0 | 18 (ambiguous) | Evaluated for safe `UNKNOWN` / `AMBIGUOUS` abstention |
| `vl_p1c/final_holdout_full` | Historical 5-class regression benchmark | 300 | 20 (`object_01`) + 80 (`object_02`–`object_05`) | 200 | Regression verification against baseline `C_v1c_cal_only` |

## 2. Hard-Negative Composition (`dev_cal_v2` vs `final_holdout_v2`)

Negatives emphasize visually similar thin, elongated, handheld, and white linear distractors:

| Negative Group (`confuser_group`) | `dev_cal_v2` (`n=124`) | `final_holdout_v2` (`n=129`) |
|---|---|---|
| `walking_stick_confuser` | 39 | 19 |
| `trekking_pole_confuser` | 4 | 20 |
| `umbrella_handle_confuser` | 35 | 0 |
| `umbrella_confuser` | 4 | 21 |
| `metal_rod_pipe_confuser` | 4 | 19 |
| `white_line_pole_confuser` | 4 | 26 |
| `general_distractor` | 34 | 24 |
| **Total Negatives** | **124** | **129** |

## 3. Metadata Schema & Zero-Leakage Isolation

- **Per-Record Fields (`manifests/dataset_manifest.json`):** Each record captures `id`, `label_id`, `confuser_group`, `sha256`, `ahash`, `dhash`, `width`, `height`, `source_title`, `source_cluster`, `source_url`, `license`, `author`, `roi_bbox_xywh`, `has_person`, `origin_phase`, and `rel_path`.
- **Provenance & Licensing:** Records in the newly curated evaluation splits (`dev_cal_v2`, `final_holdout_v2`, `ambiguous_cane_v2`) carry verified `sha256`, `source_title`, and `license` fields, whereas inherited VL-P1 gallery splits (`train_pos`, `hard_neg_gallery`) record `origin_phase: "historical"` with empty `license`.
- **Zero-Leakage Audit:** `manifests/dataset_manifest.json` (`zero_leakage_audit`) validates zero exact `sha256_overlaps`, `title_overlaps`, `cluster_overlaps`, and perceptual hash (`phash_overlaps`) across split pairs (including `dev_cal_v2__vs__ambiguous_cane_v2` and `final_holdout_v2__vs__ambiguous_cane_v2`, both `clean: true`), and `manifests/frozen_config_v2.json` confirms `final_holdout_touched_during_calibration: false`.
