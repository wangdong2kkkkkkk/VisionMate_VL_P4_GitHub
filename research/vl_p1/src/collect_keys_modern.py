"""Collect modern bunch-of-keys photos from Wikimedia Commons (avoiding posters/maps)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

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


def collect_modern_keys(base_dir: Path, target: int = 24) -> Path:
    out_dir = base_dir / "data" / "_staging_keys_modern"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    kept = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []

    seen_shas, seen_phashes, banned = _existing_signatures(base_dir)
    for sub in ("data/_staging", "data/_staging_cats", "data/_staging_keys", "data/_staging_keys_modern"):
        sub_p = base_dir / sub
        if sub_p.exists():
            for p in sub_p.rglob("*.jpg"):
                seen_shas.add(compute_sha256(p))
                try:
                    im = load_image_unicode(p)
                    seen_phashes.append((str(p), compute_ahash(im), compute_dhash(im)))
                except ValueError:
                    pass

    queries = [
        'intitle:"Schlüsselbund"',
        'intitle:"Trousseau de clés" OR intitle:"trousseau de cles"',
        'intitle:"House keys" OR intitle:"Set of keys" OR intitle:"Car keys"',
        'intitle:"Keys" keyring OR keychain',
        '"metal keys" keyring OR keychain',
        'intitle:"Keychain" keys',
    ]
    seen_u = {k["image_url"] for k in kept}
    seen_t = {k["title"].lower() for k in kept}
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    excludes = ("piano", "keyboard", "hoyt", "poster", "florida", "map", "clef", "music", "diagram", "svg", "blank", "logo", "padlock", "bottle", "bike", "museum", "uss bowfin")

    for q in queries:
        if len(kept) >= target:
            break
        try:
            hits = search_wikimedia(q, limit=25)
        except Exception as exc:
            print(f"[WARN] Search failed {q}: {exc}")
            continue
        for hit in hits:
            if len(kept) >= target:
                break
            tc = hit["title"].replace("File:", "").lower().replace("_", " ")
            if any(b in tc for b in banned) or any(x in tc for x in excludes):
                continue
            if hit["image_url"] in seen_u or hit["title"].lower() in seen_t or min(hit["width"], hit["height"]) < 180:
                continue
            time.sleep(0.2)
            try:
                r = sess.get(hit["image_url"], timeout=20)
                r.raise_for_status()
            except Exception:
                continue
            im = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None or min(im.shape[:2]) < 160:
                continue
            p = out_dir / f"mkey_{len(kept) + 1:03d}.jpg"
            save_image_unicode(p, im)
            sha, ah, dh = compute_sha256(p), compute_ahash(im), compute_dhash(im)
            if sha in seen_shas or any(
                hex_hamming_distance(ah, pah) <= 8 or hex_hamming_distance(dh, pdh) <= 8
                for _, pah, pdh in seen_phashes
            ):
                p.unlink(missing_ok=True)
                continue
            seen_shas.add(sha)
            seen_phashes.append((str(p), ah, dh))
            seen_u.add(hit["image_url"])
            seen_t.add(hit["title"].lower())
            kept.append({
                "id": f"mkey_{len(kept) + 1:03d}",
                "group": "object_02",
                "rel_path": str(p.relative_to(base_dir)).replace("\\", "/"),
                **hit,
                "width": im.shape[1],
                "height": im.shape[0],
                "sha256": sha,
                "ahash": ah,
                "dhash": dh,
            })
            print(f"  [mkey] {len(kept)}/{target}: {hit['title']}", flush=True)

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
        tx = x0 + (cw - nw) // 2
        ty = y0 + 28 + ((ch - 34) - nh) // 2
        canvas[ty:ty + nh, tx:tx + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cw - 3, y0 + ch - 3), (200, 200, 200), 1)
        cv2.putText(canvas, p.stem, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    sheet = base_dir / "manifests" / "contact_sheets_cats" / "contact_keys_modern.jpg"
    save_image_unicode(sheet, canvas)
    return sheet


if __name__ == "__main__":
    s = collect_modern_keys(Path(__file__).resolve().parent.parent)
    print("Saved modern keys contact sheet:", s)
