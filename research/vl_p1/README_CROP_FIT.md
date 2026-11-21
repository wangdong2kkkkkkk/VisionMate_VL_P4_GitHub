# VisionMate VL-P1 — ROI Fit-to-Screen helper

This patch **adds** `crop_fit.py` and `test_crop_fit.py`. It does not overwrite `orb_baseline.py`, the original photos, the VisionMate app, or any Git baseline.

## Install (from the `VisionMate_VL_P1` worktree)

```powershell
$zip = "$HOME\Downloads\VisionMate_VL_P1_Crop_Fit_Hotfix_v0_2.zip"
Test-Path $zip
Expand-Archive -LiteralPath $zip -DestinationPath .
py -3 -m unittest discover -s research\vl_p1 -p "test_*.py" -v
```

## Crop one photo first

```powershell
$src = Get-ChildItem .\research\vl_p1\data\originals\object_01 -File | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|webp)$' } | Sort-Object Name | Select-Object -First 1
py -3 .\research\vl_p1\crop_fit.py $src.FullName .\research\vl_p1\data\train\object_01\01.png
```

The photo is downscaled **only in the selection window**; the output crop uses full-resolution source pixels. Drag a rectangle around the whole cane, not just its handle. ENTER saves; ESC cancels. The helper does not overwrite existing cropped output. Then use it for remaining photos.
