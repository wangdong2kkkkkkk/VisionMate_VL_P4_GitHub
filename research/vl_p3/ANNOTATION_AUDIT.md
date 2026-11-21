# VL-P2 & VL-P3 Ground-Truth Annotation Quality Audit (Gate 0 + Repaired GT)

**Audit Artifacts:**
- Raw VL-P2 GT audit JSON: [`manifests/vl_p2_post_opt_regression_audit.json`](manifests/vl_p2_post_opt_regression_audit.json)
- VL-P3 Repaired Overrides: [`manifests/manual_cane_gt_overrides_v3.json`](manifests/manual_cane_gt_overrides_v3.json)
- Repaired Before/After Diffs: [`manifests/vl_p2_repaired_annotations.json`](manifests/vl_p2_repaired_annotations.json)
- Visual Verification Sheets:
  - [`cache/contact_sheets/gate0_vl_p2_dev_gt_audit_p1.jpg`](cache/contact_sheets/gate0_vl_p2_dev_gt_audit_p1.jpg)
  - [`cache/contact_sheets/gate0_vl_p2_dev_gt_audit_p2.jpg`](cache/contact_sheets/gate0_vl_p2_dev_gt_audit_p2.jpg)
  - [`cache/contact_sheets/gate0_vl_p2_fhold_gt_audit_p1.jpg`](cache/contact_sheets/gate0_vl_p2_fhold_gt_audit_p1.jpg)
  - [`cache/contact_sheets/gate0_vl_p2_fhold_gt_audit_p2.jpg`](cache/contact_sheets/gate0_vl_p2_fhold_gt_audit_p2.jpg)
  - [`cache/contact_sheets/gate0_vl_p2_flagged_gt_audit.jpg`](cache/contact_sheets/gate0_vl_p2_flagged_gt_audit.jpg)
  - [`cache/contact_sheets/gate0_finegrid_p1.jpg`](cache/contact_sheets/gate0_finegrid_p1.jpg), [`gate0_finegrid_p2.jpg`](cache/contact_sheets/gate0_finegrid_p2.jpg)
  - [`cache/contact_sheets/gate0_repaired_gt_verified.jpg`](cache/contact_sheets/gate0_repaired_gt_verified.jpg)

---

## 1. Initial VL-P2 Annotation Inventory (`116` Total `object_01` Positives)

Every annotation in `VisionMate_VL_P2_AG/research/vl_p2/manifests/annotation_manifest.json` (`65` in `dev_cal_v2` + `51` in `final_holdout_v2`) was audited against its original un-overlaid image by computing raw pixel shaft evidence (`cl_support_frac`, radial contrast `cl_contrast_frac`, mask background bleed `mask_bg_bleed`) **and** visually inspecting all 5 side-by-side contact sheets:

| GT Quality Tier (Pre-Repair) | Count | Source Origin Breakdown |
| :--- | :--- | :--- |
| `VERIFIED_MANUAL_GEOMETRY` | **47** | `15` `dev_cal_v2` / `51` `final_holdout_v2` manual overrides passing geometric checks |
| `PSEUDO_LABEL_HINT_ASSISTED_CLEAN` | **46** | Hint-bbox pseudo-labels in `dev_cal_v2` aligned cleanly along visible shaft |
| `MANUAL_OVERRIDE_NEEDS_MASK_OR_LINE_REFINEMENT` | **19** | Manual overrides flagged for background bleed or weak centerline support |
| `PSEUDO_LABEL_NEEDS_REPAIR` | **4** | `test_object_01_06` (`hint_bbox_diagonal_fallback`), `fhold_object_01_20` (75px stub), `fhold_object_01_06`, `hold_object_01_11` |
| **Total Flagged Automatically** | **23** | Plus **4** extra misaligned/atypical cases surfaced on full visual review = **27 repaired** |

---

## 2. Critical Annotation Defects Found & Visually Repaired (`27` Images)

Visual review with pixel coordinate grids exposed severe alignment errors in `manual_cane_gt_overrides.json` and `hint_bbox_assisted` labels:

| ID | Split | Defect Observed on Original Image | Old -> New Support / Contrast / Bleed |
| :--- | :--- | :--- | :--- |
| `fhold_object_01_13` | `dev_cal_v2` | Escalator image: old line sat on black right handrail (`x=609.6`), missing true long diagonal cane `[298,625]->[570,1320]` | `cl: 0.00->0.44`, `ctr: 0.28->0.96`, `bleed: 1.00->0.76` |
| `test_object_01_06` | `dev_cal_v2` | Z-folded cane used full-image `hint_bbox_diagonal_fallback` across black void | `cl: 0.12->1.00`, `bleed: 0.24->0.01` |
| `fhold_object_01_20` | `dev_cal_v2` | Annotated only a `75 px` diagonal stub `[136,59]->[82,113]` on `2262 px` tall cane; repaired to full vertical shaft `[358,120]->[358,2235]` | `bleed: 0.41->0.19` (full 2115px shaft) |
| `cal1c_object_01_03` | `dev_cal_v2` | Old line tilted down-right onto hedge (`[506,301]->[533,596]`), opposite to true down-left white cane `[564,358]->[513,612]` | `cl: 0.44->0.92` |
| `fhold_object_01_04` | `dev_cal_v2` | Old line drifted left into bench (`[312,325]->[235,561]`) away from vertical red-tipped cane `[355,345]->[372,560]` | `cl: 0.44->0.56` |
| `v2_ov3_039` | `final_holdout_v2` | Guide dog image: old line ran vertically between legs (`[432,269]->[336,544]`), missing cane held in right hand `[346,288]->[286,504]` | `cl: 0.04->0.60`, `bleed: 0.98->0.38` |
| `v2_ov3_067` | `final_holdout_v2` | Old line shifted ~70 px right onto empty floor (`cl=0.00`); aligned onto wrist-to-floor white cane `[400,440]->[542,656]` | `cl: 0.00->0.08`, `ctr: 0.40->0.64` (visually exact) |
| `v2_ov3_070` | `final_holdout_v2` | Old line missed upper half behind valet sign; aligned `[56,534]->[296,756]` | `cl: 0.28->0.52`, `ctr: 0.40->0.76` |
| `v2_scane_018` | `final_holdout_v2` | Museum image: old line sat behind red velvet rope `[514,396]->[667,475]` vs true cane at knee level `[448,452]->[612,578]` | `cl: 0.16->0.84`, `bleed: 1.00->0.13` |
| `v2_ovc_020` | `final_holdout_v2` | Old line skewed across left leg; aligned along vertical shaft `[190,199]->[210,584]` | `cl: 0.12->0.80` |
| `v2_fcane_003` | `final_holdout_v2` | Crowd image: steep wrong slope; aligned `[202,718]->[5,878]` | `cl: 0.40->0.96` |
| `v2_scane_015` | `final_holdout_v2` | Aligned `[244,70]->[237,580]` with tighter corridor | `cl: 0.60->0.96` |
| `v2_ov3_052` | `final_holdout_v2` | Followed true diagonal `[108,516]->[4,922]` to frame edge | `cl: 0.72->1.00` |

### Summary Improvement Across All `27` Repaired Annotations
- Mean white/red centerline support (`cl_support_frac`): **`0.333 -> 0.544`** (`+0.211`)
- Mean radial contrast support (`cl_contrast_frac`): **`0.596 -> 0.679`** (`+0.083`)
- Mean background mask bleed (`mask_bg_bleed`): **`0.653 -> 0.562`** (`-0.091`)

**Atypical Outliers Documented Honestly:**
- `cal1c_object_01_10` (`dev_cal_v2`): dark-patterned customized mobility cane (`Musician_Lachi_wearing_a_faux_fur_coat.jpg`) — geometry aligned (`cl=0.76`), flagged as atypical non-white shaft.
- `v2_ov3_085` (`final_holdout_v2`): severe motion-blurred, low-contrast cane in dark wet street (`cl=0.28`, `ctr=0.64`) — kept unmodified in historical `final_holdout_v2` split parity, repaired line aligned `[405,528]->[386,730]`.
- `v2_scane_021`, `v2_ov3_050`, `v2_ov3_061`: ultra-thin distant low-contrast shafts (`2–4 px` wide) where strict LAB white thresholds (`L* >= 58`) undercount grey/sun-glared pixels despite high radial contrast (`0.72–0.88`). Visual check on `gate0_repaired_gt_verified.jpg` confirmed centerline alignment.
