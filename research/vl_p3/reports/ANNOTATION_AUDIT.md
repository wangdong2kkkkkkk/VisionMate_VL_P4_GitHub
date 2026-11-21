# VL-P2 & VL-P3 Ground-Truth Annotation Quality Audit (Gate 0 + Repaired GT)

**Audit Artifacts:**
- Raw VL-P2 GT audit JSON: [`../manifests/vl_p2_post_opt_regression_audit.json`](../manifests/vl_p2_post_opt_regression_audit.json)
- VL-P3 Repaired Overrides: [`../manifests/manual_cane_gt_overrides_v3.json`](../manifests/manual_cane_gt_overrides_v3.json)
- Repaired Before/After Diffs: [`../manifests/vl_p2_repaired_annotations.json`](../manifests/vl_p2_repaired_annotations.json)
- Visual Verification Sheet: [`../cache/contact_sheets/gate0_repaired_gt_verified.jpg`](../cache/contact_sheets/gate0_repaired_gt_verified.jpg)

---

## 1. Initial VL-P2 Annotation Inventory (`116` Total `object_01` Positives)

Every annotation in `VisionMate_VL_P2_AG/research/vl_p2/manifests/annotation_manifest.json` (`65` in `dev_cal_v2` + `51` in `final_holdout_v2`) was audited against its original image:

| GT Quality Tier (Pre-Repair) | Count | Notes |
| :--- | :--- | :--- |
| `VERIFIED_MANUAL_GEOMETRY` | **47** | Manual overrides passing automated geometry checks |
| `PSEUDO_LABEL_HINT_ASSISTED_CLEAN` | **46** | Hint-bbox pseudo-labels in `dev_cal_v2` aligned cleanly along visible shaft |
| `MANUAL_OVERRIDE_NEEDS_MASK_OR_LINE_REFINEMENT` | **19** | Manual overrides flagged for background bleed or weak centerline support |
| `PSEUDO_LABEL_NEEDS_REPAIR` | **4** | `test_object_01_06`, `fhold_object_01_20`, `fhold_object_01_06`, `hold_object_01_11` |
| **Total Flagged Automatically / Repaired** | **23 / 27** | 4 additional imprecise/atypical cases surfaced via full contact-sheet inspection |

---

## 2. Repaired Annotation Outcomes (`27` Records Tagged `REPAIRED_VERIFIED_GEOMETRY`)

- **Mean white/red centerline support (`cl_support_frac`):** `0.333 -> 0.544` (`+0.211`)
- **Mean radial contrast support (`cl_contrast_frac`):** `0.596 -> 0.679` (`+0.083`)
- **Mean background mask bleed (`mask_bg_bleed`):** `0.653 -> 0.562` (`-0.091`)

Representative critical fixes verified on `gate0_finegrid_p1/p2.jpg` and `gate0_repaired_gt_verified.jpg`:
- `fhold_object_01_13`: moved off black escalator rail onto true diagonal cane (`cl: 0.00->0.44`, `ctr: 0.28->0.96`).
- `test_object_01_06`: replaced `hint_bbox_diagonal_fallback` across black void (`cl: 0.12->1.00`, `bleed: 0.24->0.01`).
- `fhold_object_01_20`: replaced 75px handle stub with full-length `2115px` vertical cane (`bleed: 0.41->0.19`).
- `v2_ov3_039`: moved off space between legs onto true handheld white cane (`cl: 0.04->0.60`, `bleed: 0.98->0.38`).
- `v2_scane_018`: moved off background velvet rope onto knee-level cane (`cl: 0.16->0.84`, `bleed: 1.00->0.13`).
- `v2_ovc_020`, `v2_fcane_003`, `v2_scane_015`, `v2_ov3_052`, `cal1c_object_01_03`, `v2_ov3_070`, `v2_ov3_067`: all visually verified aligned along the true cane shaft.
