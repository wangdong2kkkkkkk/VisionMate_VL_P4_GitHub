"""VisionMate VL-P1: isolated, offline ORB research baseline.

This is ROI-level instance matching, NOT open-world object detection.
No production imports, camera access, networking, or writes outside this folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TRAIN = DATA / "train"
TEST = DATA / "test"
UNKNOWN = TEST / "unknown"
IDS = [f"object_{i:02d}" for i in range(1, 6)]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Initial conservative heuristics. NOT calibrated or validated for accessibility use.
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.45
MIN_MARGIN_INLIERS = 4


def files_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT) if folder.exists() else []


def load_image(path: Path) -> np.ndarray:
    """Read Unicode paths reliably on Windows."""
    if not path.is_file():
        raise ValueError(f"Missing image: {path}")
    try:
        blob = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"Cannot open image {path}: {exc}") from exc
    if img is None or img.size == 0:
        raise ValueError(f"Invalid image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    ext = path.suffix.lower()
    if ext not in EXT:
        raise ValueError("Image output must use .jpg/.png/.bmp/.webp")
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Image encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def signature(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def setup() -> None:
    for part in (TRAIN, TEST):
        for oid in IDS:
            (part / oid).mkdir(parents=True, exist_ok=True)
    UNKNOWN.mkdir(parents=True, exist_ok=True)
    labels = HERE / "labels.json"
    if not labels.exists():
        labels.write_text(json.dumps({oid: oid for oid in IDS}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created dataset directories in: {DATA}")
    print("Put 5 cropped photos per object in train/object_XX, 10 new photos per object in test/object_XX,")
    print("and >=20 negative images in test/unknown. No images were generated or downloaded.")


def dataset_paths() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    return ({oid: files_in(TRAIN / oid) for oid in IDS},
            {oid: files_in(TEST / oid) for oid in IDS} | {"unknown": files_in(UNKNOWN)})


def inspect_dataset() -> tuple[bool, list[str]]:
    train, test = dataset_paths()
    errors = []
    for oid in IDS:
        if len(train[oid]) < 5:
            errors.append(f"{oid}: train {len(train[oid])}/5")
        if len(test[oid]) < 10:
            errors.append(f"{oid}: test {len(test[oid])}/10")
    if len(test["unknown"]) < 20:
        errors.append(f"unknown: test {len(test['unknown'])}/20")
    hashes: dict[str, set[str]] = {}
    for group, paths in (("train", sum(train.values(), [])), ("test", sum(test.values(), []))):
        for path in paths:
            h = signature(path)
            hashes.setdefault(h, set()).add(group)
    duplicates = sum(1 for splits in hashes.values() if len(splits) > 1)
    if duplicates:
        errors.append(f"LEAKAGE: {duplicates} byte-identical photos exist in both train and test")
    print("Dataset counts:")
    for oid in IDS:
        print(f"  {oid}: train={len(train[oid])}, test={len(test[oid])}")
    print(f"  unknown: test={len(test['unknown'])}")
    print(f"  duplicate SHA256 across train/test: {duplicates}")
    if errors:
        print("Not ready for final evaluation: " + "; ".join(errors))
    else:
        print("Required minimum counts and SHA256 split checks passed.")
    print("Reminder: same photo with different crop/resize can still leak; inspect shooting sessions manually.")
    return not errors, errors


def extract_orb(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1300, fastThreshold=10)
    keypoints, desc = orb.detectAndCompute(gray, None)
    return keypoints, desc


def match_pair(a, b) -> dict[str, float | int]:
    kp_a, desc_a = a
    kp_b, desc_b = b
    empty = {"good": 0, "inliers": 0, "ratio": 0.0}
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return empty
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return {"good": len(good), "inliers": 0, "ratio": 0.0}
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        n = int(mask.sum()) if mask is not None else 0
    except cv2.error:
        n = 0
    return {"good": len(good), "inliers": n, "ratio": round(n / len(good), 4)}


def build_gallery() -> dict[str, list[tuple[Path, tuple]]]:
    train, _ = dataset_paths()
    gallery = {}
    for oid, paths in train.items():
        gallery[oid] = [(path, extract_orb(load_image(path))) for path in paths]
    if not any(gallery.values()):
        raise ValueError("No training images. Run 'setup' then add cropped photos.")
    return gallery


def predict_one(image: np.ndarray, gallery: dict) -> dict:
    query = extract_orb(image)
    ranked = []
    for oid in IDS:
        scores = [(match_pair(feat, query), path.name) for path, feat in gallery.get(oid, [])]
        if scores:
            match, exemplar = max(scores, key=lambda item: (item[0]["inliers"], item[0]["good"]))
            ranked.append({"id": oid, "sample": exemplar, **match})
    ranked.sort(key=lambda row: (row["inliers"], row["good"]), reverse=True)
    if not ranked:
        return {"status": "UNKNOWN", "label": None, "candidates": []}
    top = ranked[0]
    second = ranked[1]["inliers"] if len(ranked) > 1 else 0
    if top["inliers"] < MIN_INLIERS or top["ratio"] < MIN_INLIER_RATIO:
        status = "UNKNOWN"
    elif top["inliers"] - second < MIN_MARGIN_INLIERS:
        status = "AMBIGUOUS"
    else:
        status = "KNOWN"
    return {"status": status, "label": top["id"] if status == "KNOWN" else None,
            "candidates": ranked[:3]}


def cmd_predict(path: Path) -> None:
    output = predict_one(load_image(path), build_gallery())
    labels_path = HERE / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    if output["label"]:
        output["name"] = labels.get(output["label"], output["label"])
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("Experimental score only; not a calibrated confidence/probability.")


def cmd_evaluate() -> None:
    ready, _ = inspect_dataset()
    if not ready:
        raise SystemExit("Evaluation blocked: incomplete dataset or exact duplicate leakage.")
    gallery = build_gallery()
    _, tests = dataset_paths()
    correct, unknown_correct, errors = 0, 0, []
    confusion = Counter()
    n_pos = sum(len(tests[oid]) for oid in IDS)
    for true in [*IDS, "unknown"]:
        for image_path in tests[true]:
            pred = predict_one(load_image(image_path), gallery)
            got = pred["label"] or pred["status"]
            confusion[f"{true}->{got}"] += 1
            if true != "unknown":
                correct += int(got == true)
            else:
                unknown_correct += int(pred["status"] == "UNKNOWN")
            if (true != "unknown" and got != true) or (true == "unknown" and pred["status"] == "KNOWN"):
                errors.append({"image": str(image_path.relative_to(HERE)), "ground_truth": true,
                               "prediction": got, "top_candidates": pred["candidates"]})
    summary = {"train_images": sum(map(len, dataset_paths()[0].values())),
               "test_known_images": n_pos, "test_unknown_images": len(tests["unknown"]),
               "correct_known": correct, "unknown_rejected": unknown_correct,
               "known_accuracy": round(correct / n_pos, 4) if n_pos else None,
               "false_accept_rate_unknown": round(1 - unknown_correct / len(tests["unknown"]), 4),
               "confusion": dict(sorted(confusion.items())), "errors": errors,
               "note": "ORB ROI-only prototype; thresholds are provisional; not independently validated."}
    report_path = HERE / "reports" / "orb_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("errors", "confusion")}, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")


def cmd_crop(src: Path, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"Refusing to overwrite existing crop: {dst}")
    frame = load_image(src)
    print("Drag a rectangle around ONE object. Press ENTER to confirm or ESC to cancel.")
    try:
        box = cv2.selectROI("VisionMate VL-P1 | Draw object ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        raise SystemExit(f"OpenCV ROI window unavailable: {exc}") from exc
    x, y, w, h = map(int, box)
    if w <= 0 or h <= 0:
        print("Crop canceled; no file saved.")
        return
    save_image(dst, frame[y:y+h, x:x+w])
    print(f"Saved crop: {dst} ({w}x{h})")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionMate offline ORB research only")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    sub.add_parser("status")
    pred = sub.add_parser("predict")
    pred.add_argument("image", type=Path)
    crop = sub.add_parser("crop")
    crop.add_argument("source", type=Path)
    crop.add_argument("output", type=Path)
    sub.add_parser("evaluate")
    args = parser.parse_args()
    if args.command == "setup": setup()
    elif args.command == "status": inspect_dataset()
    elif args.command == "predict": cmd_predict(args.image)
    elif args.command == "crop": cmd_crop(args.source, args.output)
    elif args.command == "evaluate": cmd_evaluate()


if __name__ == "__main__":
    main()
