# VL-P1b Confusion Matrices

### ROI Holdout — A0_baseline_hybrid

| GT \ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS |
|---|---|---|---|---|---|---|---|
| `object_01` | 10 | 0 | 0 | 0 | 0 | 10 | 0 |
| `object_02` | 0 | 17 | 0 | 0 | 0 | 3 | 0 |
| `object_03` | 0 | 0 | 12 | 0 | 0 | 8 | 0 |
| `object_04` | 0 | 0 | 0 | 11 | 2 | 7 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 17 | 3 | 0 |
| `unknown` | 3 | 2 | 3 | 0 | 0 | 106 | 0 |

### ROI Holdout — A3_full_verifier

| GT \ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS |
|---|---|---|---|---|---|---|---|
| `object_01` | 11 | 0 | 0 | 0 | 0 | 9 | 0 |
| `object_02` | 0 | 17 | 0 | 0 | 0 | 3 | 0 |
| `object_03` | 0 | 0 | 11 | 0 | 0 | 9 | 0 |
| `object_04` | 0 | 0 | 0 | 16 | 2 | 2 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 19 | 1 | 0 |
| `unknown` | 5 | 13 | 1 | 1 | 0 | 94 | 0 |

### E2E Holdout — C1_vl_p1_baseline

| GT \ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---|---|---|---|---|---|---|---|
| `object_01` | 5 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| `object_02` | 0 | 18 | 0 | 0 | 0 | 2 | 0 | 0 |
| `object_03` | 0 | 0 | 11 | 0 | 0 | 9 | 0 | 0 |
| `object_04` | 0 | 0 | 0 | 14 | 1 | 5 | 0 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 15 | 5 | 0 | 0 |
| `unknown` | 6 | 3 | 4 | 2 | 0 | 99 | 0 | 0 |

### E2E Holdout — C2_hard_neg_only

| GT \ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---|---|---|---|---|---|---|---|
| `object_01` | 6 | 0 | 0 | 0 | 0 | 14 | 0 | 0 |
| `object_02` | 0 | 18 | 0 | 0 | 0 | 2 | 0 | 0 |
| `object_03` | 0 | 1 | 11 | 0 | 0 | 8 | 0 | 0 |
| `object_04` | 1 | 0 | 0 | 17 | 1 | 1 | 0 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 18 | 2 | 0 | 0 |
| `unknown` | 9 | 16 | 1 | 3 | 2 | 83 | 0 | 0 |

### E2E Holdout — C3_proposal_only

| GT \ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---|---|---|---|---|---|---|---|
| `object_01` | 3 | 0 | 0 | 0 | 1 | 16 | 0 | 0 |
| `object_02` | 0 | 18 | 0 | 0 | 0 | 2 | 0 | 0 |
| `object_03` | 0 | 0 | 10 | 0 | 0 | 10 | 0 | 0 |
| `object_04` | 1 | 0 | 0 | 13 | 0 | 6 | 0 | 0 |
| `object_05` | 0 | 0 | 0 | 0 | 16 | 4 | 0 | 0 |
| `unknown` | 8 | 1 | 6 | 2 | 0 | 97 | 0 | 0 |

### E2E Holdout — C4_combined_v1b

| GT \ Pred | object_01 | object_02 | object_03 | object_04 | object_05 | UNKNOWN | AMBIGUOUS | PROPOSAL_MISSED |
|---|---|---|---|---|---|---|---|---|
| `object_01` | 7 | 0 | 0 | 0 | 1 | 12 | 0 | 0 |
| `object_02` | 0 | 17 | 0 | 0 | 0 | 3 | 0 | 0 |
| `object_03` | 1 | 1 | 9 | 0 | 0 | 9 | 0 | 0 |
| `object_04` | 1 | 0 | 0 | 16 | 1 | 2 | 0 | 0 |
| `object_05` | 1 | 0 | 0 | 0 | 17 | 2 | 0 | 0 |
| `unknown` | 15 | 15 | 4 | 3 | 1 | 76 | 0 | 0 |
