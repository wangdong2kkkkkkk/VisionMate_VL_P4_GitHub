"""Collect real bunch-of-keys / keyring photography supplements from Wikimedia Commons."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from .collect_category_supplements import fetch_category_members
from .collect_dataset import USER_AGENT, _existing_signatures, search_wikimedia
from .image_utils import (
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
    save_image_unicode,
)

KEY_SOURCES = [
    ("cat", "Category:Keys"),
    ("cat", "Category:Key_fobs"),
    ("search", 'intitle:"Schlüsselbund" OR intitle:"Schluesselbund"'),
    ("search", 'intitle:"trousseau de clés" OR intitle:"trousseau de cles"'),
    ("search", 'intitle:"house keys" OR intitle:"set of keys" OR intitle:"keys on"'),
    ("search", '"bunch of metallic keys" OR "key bunch" OR "metal keys" keyring'),
    ("search", 'intitle:"car key" OR intitle:"car keys"'),
]


def collect_keys(base_dir: Path, target: int = 25, sleep_sec: float = 0.25) -> Path:
    out_dir = base_dir / "data" / "_staging_keys"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    kept = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []

    seen_shas, seen_phashes, banned_titles = _existing_signatures(base_dir)
    for sub in ("data/_staging", "data/_staging_cats", "data/_staging_keys"):
        p_root = base_dir / sub
        if p_root.exists():
            for p in p_root.rglob("*.jpg"):
                seen_shas.add(compute_sha256(p))
                try:
                    im = load_image_unicode(p)
                    seen_phashes.append((str(p), compute_ahash(im), compute_dhash(im)))
                except ValueError:
                    pass

    seen_urls = {k["image_url"] for k in kept}
    seen_titles = {k["title"].lower() for k in kept}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    excludes = ("piano", "keyboard", "hoyt", "poster", "florida", "map", "clef", "music", "coin", "padlock", "locksmith sign", "diagram", "svg")

    for kind, src in KEY_SOURCES:
        if len(kept) >= target:
            break
        try:
            hits = fetch_category_members(src, limit=40) if kind == "cat" else search_wikimedia(src, limit=30)
        except Exception as exc:
            print(f"[WARN] Failed {kind} {src}: {exc}")
            continue
        for hit in hits:
            if len(kept) >= target:
                break
            tc = hit["title"].replace("File:", "").lower().replace("_", " ")
            if any(b in tc for b in banned_titles):
                continue
            if any(ex in tc for ex in excludes):
                continue
            if hit["image_url"] in seen_urls or hit["title"].lower() in seen_titles:
                continue
            if min(hit["width"], hit["height"]) < 180:
                continue

            time.sleep(sleep_sec)
            try:
                r = session.get(hit["image_url"], timeout=25)
                r.raise_for_status()
            except Exception:
                continue
            img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            ih, iw = img.shape[:2]
            if min(ih, iw) < 160 or max(ih / max(1, iw), iw / max(1, ih)) > 4.0:
                continue

            idx = len(kept) + 1
            out_path = out_dir / f"key_{idx:03d}.jpg"
            save_image_unicode(out_path, img)
            sha = compute_sha256(out_path)
            ah = compute_ahash(img)
            dh = compute_dhash(img)
            if sha in seen_shas or any(
                hex_hamming_distance(ah, pah) <= 8 or hex_hamming_distance(dh, pdh) <= 8
                for _, pah, pdh in seen_phashes
            ):
                out_path.unlink(missing_ok=True)
                continue

            rel = str(out_path.relative_to(base_dir)).replace("\\", "/")
            seen_shas.add(sha)
            seen_phashes.append((rel, ah, dh))
            seen_urls.add(hit["image_url"])
            seen_titles.add(hit["title"].lower())
            kept.append({
                "id": f"object_02_key_{idx:03d}",
                "group": "object_02",
                "rel_path": rel,
                **hit,
                "width": iw,
                "height": ih,
                "sha256": sha,
                "ahash": ah,
                "dhash": dh,
            })
            print(f"  [object_02] {len(kept)}/{target}: {hit['title']} ({iw}x{ih})", flush=True)

    meta_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")

    imgs = list_images(out_dir)
    cols = 5
    cw, ch = 260, 220
    rows = max(1, (len(imgs) + cols - 1) // cols)
    canvas = np.full((rows * ch, cols * cw, 3), 245, dtype=np.uint8)
    for i, p in enumerate(imgs):
        r, c = divmod(i, cols)
        x0, y0 = c * cw, r * ch
        im = load_image_unicode(p)
        ih, iw = im.shape[:2]
        sc = min((cw - 16) / iw, (ch - 38) / ih)
        nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
        th = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        tx = x0 + (cw - nw) // 2
        ty = y0 + 28 + ((ch - 34) - nh) // 2
        canvas[ty:ty + nh, tx:tx + nw] = th
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cw - 3, y0 + ch - 3), (200, 200, 200), 1)
        cv2.putText(canvas, p.stem, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    sheet_path = base_dir / "manifests" / "contact_sheets_cats" / "contact_keys_supplement.jpg"
    save_image_unicode(sheet_path, canvas)
    return sheet_path


if __name__ == "__main__":
    p = collect_keys(Path(__file__).resolve().parent.parent)
    print("Saved keys contact sheet:", p)
