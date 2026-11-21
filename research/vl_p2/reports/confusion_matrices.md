# End-to-End Confusion Matrices & Hypothesis Testing — VL-P2

All numbers below are copied verbatim from `research/vl_p2/reports/benchmark_metrics.json`.

## 1. Blind Holdout Matrix (`final_holdout_v2`, `N = 180`: `51` `object_01`, `129` Negatives)

| Preset | Overall Acc (95% CI) | `object_01` Recall (95% CI) | Neg FAR (95% CI) | Neg Specificity | Ambiguous Rate (`pos`/`neg`) | 3-Way CM `{object_01 / unknown / ambiguous}` |
|---|---|---|---|---|---|---|
| `A0_axis_baseline` | **148/180 (82.2%)** `[75.98, 87.12]` | 22/51 (43.1%) `[30.50, 56.73]` | **3/129 (2.3%)** `[0.79, 6.61]` | **126/129 (97.7%)** | 43.33% (`20`/`58`) | Pos `22/9/20` \| Neg `3/68/58` |
| `A1_oriented_rotated` | 129/180 (71.7%) | 10/51 (19.6%) | 10/129 (7.8%) | 119/129 (92.2%) | 58.33% (`29`/`76`) | Pos `10/12/29` \| Neg `10/43/76` |
| `B1_seg_masked_axis` (`rep3`) | 140/180 (77.8%) | 21/51 (41.2%) `[28.75, 54.83]` | 10/129 (7.8%) `[4.26, 13.68]` | 119/129 (92.2%) | 68.89% (`28`/`96`) | Pos `21/2/28` \| Neg `10/23/96` |
| `B2_oriented_seg_rot_masked` (`rep4`) | 122/180 (67.8%) | 6/51 (11.8%) | 13/129 (10.1%) | 116/129 (89.9%) | 89.44% (`45`/`116`) | Pos `6/0/45` \| Neg `13/0/116` |
| `C1_best_rep_calibrated` (`rep3`) | 140/180 (77.8%) | 21/51 (41.2%) `[28.75, 54.83]` | 10/129 (7.8%) `[4.26, 13.68]` | 119/129 (92.2%) | 68.89% (`28`/`96`) | Pos `21/2/28` \| Neg `10/23/96` |
| **`C2_full_cane_verified`** (`rep3` + Verifier) | 145/180 (80.6%) `[74.16, 85.67]` | **30/51 (58.8%)** `[45.17, 71.25]` | 14/129 (10.8%) `[6.58, 17.39]` | 115/129 (89.2%) | 62.22% (`19`/`93`) | Pos `30/2/19` \| Neg `14/22/93` |

## 2. McNemar Paired Significance Tests vs `A0_axis_baseline`

| Comparison vs `A0` | All Samples (`N=180`) | Positive Recall (`n=51`) | Negative Specificity (`n=129`) |
|---|---|---|---|
| `A1_oriented_rotated` | `-10.56 pp` (`p = 0.000546`) | `-23.53 pp` (`p = 0.007538`) | `-5.43 pp` (`p = 0.06543`) |
| `B1_seg_masked_axis` / `C1` | `-4.44 pp` (`p = 0.115318`) | `-1.96 pp` (`p = 1.000000`) | `-5.43 pp` (`p = 0.039062`) |
| `B2_oriented_seg_rot_masked` | `-14.44 pp` (`p = 0.000003`) | `-31.37 pp` (`p = 0.000145`) | — |
| **`C2_full_cane_verified`** | `-1.67 pp` `[-7.22, +3.89]` (`p = 0.701108`) | **`+15.69 pp`** `[+1.96, +29.41]` (`p = 0.057373`, **11 new wins vs 3 ref wins**) | `-8.53 pp` `[-13.95, -3.10]` (`p = 0.003418`) |

## 3. Negative Subgroup FAR & Scenario Recall (`final_holdout_v2`)

| Confuser Group / Scenario | `A0_axis_baseline` | `B1` / `C1` (`rep3`) | `C2_full_cane_verified` |
|---|---|---|---|
| `general_distractor` (`n=24`) | 0/24 (0.0%) | 0/24 (0.0%) | 1/24 (4.2%) |
| `metal_rod_pipe_confuser` (`n=19`) | 1/19 (5.3%) | 2/19 (10.5%) | 3/19 (15.8%) |
| `trekking_pole_confuser` (`n=20`) | 1/20 (5.0%) | 2/20 (10.0%) | 3/20 (15.0%) |
| `umbrella_confuser` (`n=21`) | 0/21 (0.0%) | 1/21 (4.8%) | 3/21 (14.3%) |
| `walking_stick_confuser` (`n=19`) | 0/19 (0.0%) | 2/19 (10.5%) | 3/19 (15.8%) |
| `white_line_pole_confuser` (`n=26`) | 1/26 (3.9%) | 3/26 (11.5%) | 1/26 (3.9%) |
| `handheld` positives (`n=47`) | 19/47 (40.4%) | 18/47 (38.3%) | **28/47 (59.6%)** |
| `occluded` positives (`n=28`) | 8/28 (28.6%) | 6/28 (21.4%) | **13/28 (46.4%)** |
| `red_tip` positives (`n=3`) | 2/3 (66.7%) | 1/3 (33.3%) | 2/3 (66.7%) |
| `folded` positives (`n=1`) | 1/1 (100.0%) | 1/1 (100.0%) | 0/1 (0.0%) |

## 4. Ambiguous Abstention Safety (`ambiguous_cane_v2`, `n = 18`)

| Preset | Forced `KNOWN` (`FK`) | `AMBIGUOUS` | `UNKNOWN` | Safe Abstention Rate |
|---|---|---|---|---|
| `A0_axis_baseline` | 2 | 5 | 11 | 16/18 (88.9%) |
| `A1_oriented_rotated` | 0 | 12 | 6 | 18/18 (100.0%) |
| `B1_seg_masked_axis` | 2 | 13 | 3 | 16/18 (88.9%) |
| `B2_oriented_seg_rot_masked` | 0 | 18 | 0 | 18/18 (100.0%) |
| `C1_best_rep_calibrated` | 2 | 13 | 3 | 16/18 (88.9%) |
| `C2_full_cane_verified` | 2 | 11 | 5 | 16/18 (88.9%) |

## 5. Historical 5-Class Integration (`vl_p1c/final_holdout_full`, `N = 300`)

| System | Overall Acc (95% CI) | Known Recall | Neg FAR | `object_01` (Cane) | `object_02` | `object_03` | `object_04` | `object_05` |
|---|---|---|---|---|---|---|---|---|
| Baseline `C_v1c_cal_only` | 251/300 (83.7%) `[79.06, 87.42]` | 62/100 (62.0%) | 11/200 (5.5%) | 2/20 (10.0%) | 16/20 (80.0%) | 14/20 (70.0%) | 13/20 (65.0%) | 17/20 (85.0%) |
| **`VL-P2 C2 + VL-P1c R5`** | **258/300 (86.0%)** `[81.62, 89.47]` | **73/100 (73.0%)** `[63.57, 80.73]` | 15/200 (7.5%) `[4.60, 12.00]` | **13/20 (65.0%)** `[43.29, 81.88]` | 16/20 (80.0%) | 14/20 (70.0%) | 13/20 (65.0%) | 17/20 (85.0%) |

Integrating VL-P2 `C2` for `object_01` alongside VL-P1c `R5` for `object_02`–`object_05` increased white-cane recall on the historical 5-class holdout by **+55.0 percentage points** (`10.0% -> 65.0%`) with zero regression on `object_02`–`object_05`.
