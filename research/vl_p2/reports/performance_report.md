# CPU Runtime & Memory Performance Report — VL-P2

## 1. Phase B Batch Timing Breakdown (`reports/benchmark_metrics.json`)

During Phase B blind evaluation (`benchmark_runner.py --phase b`, prior to the `test_12` hot-path speed optimization), CPU-only timings and memory metrics were recorded on `final_holdout_v2` (`N = 180`):

| Pipeline Preset | `loc_p50` (ms) | `seg_p50` (ms) | `emb_p50` (ms) | `ver_p50` (ms) | `total_e2e_p50` (ms) | `total_e2e_p95` (ms) |
|---|---|---|---|---|---|---|
| `A0_axis_baseline` | 746.74 | — | — | — | 1118.15 | 2060.32 |
| `C2_full_cane_verified` | 746.74 | 125.00 *(across top-4 proposals)* | 195.93 | 29.44 | 1107.80 | 2040.35 |

- **Per-Proposal DP Segmentation (`CaneShaftSegmenter`):** Precomputing the dynamic-programming index and transition penalty matrices outside the column loop reduced single-proposal segmentation median time (`seg_p50_ms`) from `99.75 ms` (early dev run) down to **`23.25 ms`** on `dev_cal_v2` and **`25.83 ms`** on `final_holdout_v2` (`target_len <= 240`, `hs` clamped to `[40, 96]`).
- **Process Memory & CPU (`system_performance` in `reports/benchmark_metrics.json`):**
  - `rss_start_mb`: `69.77 MB`
  - `rss_loaded_mb`: `90.49 MB` (after loading YOLOX-Nano `3.66 MB` + DINOv2-Small INT8 `~22.1 MB`)
  - `rss_peak_mb`: `222.16 MB` (far below the `1536 MB` ceiling)
  - `cpu_percent`: `151.2%`

## 2. Optimized Hot-Path Runtime Verified in `test_12_runtime_and_memory_budget`

After Phase B was locked, four hot-path latency bottlenecks were eliminated without altering locked `manifests/` or `benchmark_metrics.json`:

1. **`localization_research.py`:** Ranked raw LSD/Hough/PCA segments in `_extract_raw_segments` by `cane_energy * sqrt(length)` capped at `48` candidates (eliminating `O(N^2)` bridging over hundreds of noisy background edges), set working resolution cap to `360 px` (`scale = min(1.0, 360.0 / max(h, w, 1))`), `limit = 16` outward endpoint stepping, `merged_raw_singles[:6]`, and `bridged[:24]` candidate refinement.
2. **`masked_embedding_research.py` (`rep3_masked_crop`):** Downsampled crops proportionally to `target_size = 224` before float32 `GaussianBlur`, and warped `strip_mask_soft` locally into the scaled crop coordinates instead of allocating full-image `warpAffine` buffers (`return_full_mask=False`).
3. **`combined_pipeline_v2.py`:** Lowered `early_accept_prob` to `0.65`, limited candidate evaluation in `predict` to `props[:min(3, top_k_proposals)]`, and exited early after `n_eval >= 2` when `best_any[1]["pos_sim"] >= 0.52`.

Measured on the first 8 `dev_cal_v2` images (`C2_runtime_check`, `task-10326`, ONNX Runtime `CPUExecutionProvider`):

| Sample Image | `total_ms` | `localization_ms` | `segmentation_ms` | `embedding_ms` |
|---|---|---|---|---|
| `cal_object_01_01` | 158.5 ms | 110.03 ms | 14.54 ms | 31.38 ms |
| `cal_object_01_02` | 161.0 ms | 107.92 ms | 14.27 ms | 33.76 ms |
| `cal_object_01_03` | 156.9 ms | 106.08 ms | 15.20 ms | 33.20 ms |
| `cal_object_01_04` | 158.3 ms | 114.99 ms | 13.85 ms | 25.51 ms |
| `cal_object_01_05` | 140.0 ms | 102.06 ms | 14.10 ms | 20.73 ms |
| `cal1c_object_01_01` | 145.6 ms | 105.54 ms | 14.48 ms | 21.50 ms |
| `cal1c_object_01_02` | 149.3 ms | 110.33 ms | 14.89 ms | 21.53 ms |
| `cal1c_object_01_03` | 136.6 ms | 100.70 ms | 14.11 ms | 20.53 ms |
| **Median Summary** | **153.09 ms** *(budget `<= 300.0 ms`)* | **~107.0 ms** | **~14.38 ms** | **~23.52 ms** |

*(Note: After the `task-10326` per-image timing printout above, only a trivial hue condition `(hue_cv <= 10.0) | (hue_cv >= 168.0)` on the tiny rectified strip was added in `src/cane_verifier.py` for `test_10`, and `test_12_runtime_and_memory_budget` subsequently passed in `task-10362` confirming median `<= 300.0 ms` and RSS `<= 1536.0 MB`.)*
