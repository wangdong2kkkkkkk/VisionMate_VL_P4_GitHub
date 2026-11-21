# Multi-Geometry White-Cane Annotation Protocol — VL-P2

## 1. Scope & Coverage

`research/vl_p2/src/annotation_protocol.py` builds the verified multi-geometry annotation manifest `research/vl_p2/manifests/annotation_manifest.json` (`sha256: 30360be6a9700b3c0ce55fe3bb0c2c32545b99972c2a751a004803fc4d1838ba`) covering **116** `object_01` images (`total_annotated_object_01: 116`):

- `dev_cal_v2`: **65** `object_01` images (including occluded and handheld subsets).
- `final_holdout_v2`: **51** blind `object_01` images (`47` tagged `handheld`, `28` `occluded`, `3` `red_tip`, `1` `folded` within evaluation breakdowns).
- Diagnostic support is also provided for `historical_v1c_holdout` (`20` `object_01` images).

## 2. Six Required Multi-Geometry Fields

Every `object_01` entry in `annotation_manifest.json` stores six complementary geometric and semantic annotations:

1. `bbox_xywh`: axis-aligned bounding box `[x, y, w, h]`.
2. `obb_cxcywh_angle` + `obb_corners_xy`: tight rotated oriented bounding box.
3. `endpoints_xyxy` + `centerline_points_xy`: handle-to-tip endpoint coordinates and centerline polyline.
4. `mask_rle`: run-length encoded foreground shaft mask (`size: [H, W]`, `counts: [...]`) constrained inside the verified OBB.
5. `occluded`: boolean occlusion flag + `scenario_tags` (`handheld`, `red_tip`, `folded`, `crowded_bg`, etc.).
6. `label_id` (`object_01`, `unknown`, `ambiguous`) + `confuser_group`.

## 3. Synthesis, Manual Override & Visual QA Workflow

1. **Hint-Constrained Initialization:** When an axis-aligned `hint_bbox_xywh` is available, `src/annotation_protocol.py` expands the crop by `8%` (minimum `8 px`) and runs ridge feature mapping (`_compute_cane_feature_maps`), raw segment extraction (`_extract_raw_segments`), collinear bridging (`_bridge_collinear_segments`), and refinement (`_evaluate_and_refine_candidate`). If ridge evidence inside the box is weak, a clean diagonal initialization across the hint box is used instead.
2. **Unconstrained Proposal & Manual Overrides:** Without a hint box, `OrientedCaneLocalizer.detect_with_context(..., top_k=3)` initializes candidate endpoints, with a centered fallback placeholder flagged for manual review via `manifests/manual_cane_gt_overrides.json`.
3. **Derived Geometry & Mask Rasterization:** `derive_gt_geometry_from_endpoints` generates the oriented bounding box (`obb_from_endpoints`) and pixel shaft mask (`CaneShaftSegmenter`), while `encode_mask_rle` serializes the RLE mask.
4. **Visual Inspection Overlay:** `render_gt_overlay_preview` renders yellow shaft masks, green rotated OBB polygons, red centerlines, and cyan endpoint markers onto QA previews before locking the manifest.
