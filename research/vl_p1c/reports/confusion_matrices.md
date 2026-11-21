# Confusion Matrices — VL-P1c (Locked Blind Evaluation)

## 1. ROI Classification on `final_holdout_roi` (`N=300`)

### R0_vl_p1_baseline

| GT \\ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 11 | 0 | 0 | 0 | 0 | 9 | 0 |
| `object_02` | 0 | 14 | 0 | 0 | 0 | 6 | 0 |
| `object_03` | 0 | 0 | 13 | 0 | 0 | 7 | 0 |
| `object_04` | 0 | 0 | 0 | 10 | 0 | 10 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| `unknown` | 7 | 4 | 4 | 0 | 1 | 184 | 0 |

### A3_vl_p1b_ref (VL-P1b reference)

| GT \\ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 14 | 0 | 0 | 0 | 0 | 6 | 0 |
| `object_02` | 0 | 15 | 0 | 0 | 0 | 5 | 0 |
| `object_03` | 0 | 0 | 8 | 0 | 0 | 12 | 0 |
| `object_04` | 0 | 0 | 0 | 16 | 0 | 4 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| `unknown` | 12 | 2 | 2 | 0 | 1 | 183 | 0 |

### R5_selective_3state_v1c (VL-P1c Locked)

| GT \\ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 10 | 0 | 0 | 0 | 0 | 8 | 2 |
| `object_02` | 0 | 13 | 0 | 0 | 0 | 5 | 2 |
| `object_03` | 0 | 0 | 10 | 0 | 0 | 9 | 1 |
| `object_04` | 0 | 0 | 0 | 11 | 0 | 6 | 3 |
| `object_05` | 0 | 0 | 0 | 0 | 19 | 0 | 1 |
| `unknown` | 4 | 2 | 3 | 0 | 1 | 184 | 6 |

## 2. End-to-End Pipelines on `final_holdout_full` (`N=300`)

### C1_vl_p1_baseline

| GT \\ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 2 | 0 | 0 | 0 | 0 | 18 | 0 | 0 |
| `object_02` | 0 | 16 | 0 | 0 | 0 | 4 | 0 | 0 |
| `object_03` | 0 | 0 | 16 | 0 | 0 | 4 | 0 | 0 |
| `object_04` | 0 | 0 | 0 | 13 | 0 | 7 | 0 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 17 | 3 | 0 | 0 |
| `unknown` | 8 | 5 | 8 | 0 | 0 | 179 | 0 | 0 |

### C4_vl_p1b_combined

| GT \\ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 4 | 0 | 0 | 0 | 0 | 16 | 0 | 0 |
| `object_02` | 3 | 15 | 0 | 0 | 0 | 2 | 0 | 0 |
| `object_03` | 2 | 1 | 13 | 0 | 0 | 4 | 0 | 0 |
| `object_04` | 0 | 1 | 0 | 16 | 0 | 3 | 0 | 0 |
| `object_05` | 1 | 0 | 0 | 0 | 19 | 0 | 0 | 0 |
| `unknown` | 42 | 5 | 3 | 2 | 0 | 147 | 1 | 0 |

### C_v1c_full (Frozen K=4)

| GT \\ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `object_01` | 2 | 0 | 0 | 0 | 0 | 18 | 0 | 0 |
| `object_02` | 0 | 11 | 0 | 0 | 0 | 3 | 6 | 0 |
| `object_03` | 1 | 0 | 11 | 0 | 0 | 4 | 4 | 0 |
| `object_04` | 0 | 0 | 0 | 10 | 0 | 4 | 6 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 16 | 2 | 2 | 0 |
| `unknown` | 3 | 1 | 4 | 0 | 0 | 174 | 18 | 0 |
