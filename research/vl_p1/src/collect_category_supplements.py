"""Supplemental Wikimedia Commons Category-based downloader for verified distinct photography."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from .collect_dataset import COMMONS_API, USER_AGENT, _existing_signatures, _strip_html, build_contact_sheets
from .image_utils import (
    OBJECT_IDS,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
    save_image_unicode,
)

CATEGORY_SPECS = {
    "object_01": (["Category:White_canes", "Category:White_cane_users"], 25),
    "object_02": (["Category:Key_rings", "Category:Keys"], 25),
    "object_03": (["Category:Leather_wallets", "Category:Wallets"], 25),
    "object_04": (["Category:Slippers", "Category:Flip-flops"], 15),
    "object_05": (["Category:Running_shoes", "Category:Dress_shoes"], 15),
}


def fetch_category_members(category: str, limit: int = 40) -> list[dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 800,
        "format": "json",
    }
    r = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for _, page in sorted(pages.items(), key=lambda kv: int(kv[0])):
        title = page.get("title", "")
        iis = page.get("imageinfo", [])
        if not iis:
            continue
        ii = iis[0]
        if ii.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        meta = ii.get("extmetadata", {})
        lic = _strip_html(meta.get("LicenseShortName", {}).get("value", "Unknown"))
        if not any(k in lic.upper() for k in ("CC", "PUBLIC DOMAIN", "CC0", "PDM")):
            continue
        out.append({
            "title": title,
            "page_url": ii.get("descriptionurl", ""),
            "image_url": ii.get("thumburl") or ii.get("url"),
            "orig_url": ii.get("url", ""),
            "width": ii.get("thumbwidth") or ii.get("width", 0),
            "height": ii.get("thumbheight") or ii.get("height", 0),
            "license": lic,
            "author": _strip_html(meta.get("Artist", {}).get("value", "Unknown"))[:120],
            "description": _strip_html(meta.get("ImageDescription", {}).get("value", ""))[:250],
            "query": category,
        })
    return out


def download_category_supplements(base_dir: Path, sleep_sec: float = 0.25) -> None:
    staging_cats = base_dir / "data" / "_staging_cats"
    staging_cats.mkdir(parents=True, exist_ok=True)
    seen_shas, seen_phashes, banned_titles = _existing_signatures(base_dir)

    # Also register hashes already present in data/_staging to prevent duplicates
    old_staging = base_dir / "data" / "_staging"
    if old_staging.exists():
        for p in old_staging.rglob("*.jpg"):
            seen_shas.add(compute_sha256(p))
            try:
                img = load_image_unicode(p)
                seen_phashes.append((str(p), compute_ahash(img), compute_dhash(img)))
            except ValueError:
                pass

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for group, (cats, target) in CATEGORY_SPECS.items():
        gdir = staging_cats / group
        gdir.mkdir(parents=True, exist_ok=True)
        meta_path = gdir / "metadata.json"
        kept = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
        seen_urls = {k["image_url"] for k in kept}
        seen_titles = {k["title"].lower() for k in kept}

        for cat in cats:
            if len(kept) >= target:
                break
            try:
                hits = fetch_category_members(cat, limit=45)
            except Exception as exc:
                print(f"[WARN] Category failed {cat}: {exc}")
                continue
            for hit in hits:
                if len(kept) >= target:
                    break
                tc = hit["title"].replace("File:", "").lower().replace("_", " ")
                if any(b in tc for b in banned_titles):
                    continue
                if any(bad in tc for bad in ("logo", "icon", "diagram", "coin purse?no", "piano", "bitcoin", "crypto")):
                    continue
                if hit["image_url"] in seen_urls or hit["title"].lower() in seen_titles:
                    continue
                if min(hit["width"], hit["height"]) < 180:
                    continue

                time.sleep(sleep_sec)
                try:
                    resp = session.get(hit["image_url"], timeout=25)
                    resp.raise_for_status()
                except Exception:
                    continue
                img = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue
                ih, iw = img.shape[:2]
                if min(ih, iw) < 160 or max(ih / max(1, iw), iw / max(1, ih)) > 4.5:
                    continue

                idx = len(kept) + 1
                out_path = gdir / f"cat_{idx:03d}.jpg"
                save_image_unicode(out_path, img)
                sha = compute_sha256(out_path)
                ah = compute_ahash(img)
                dh = compute_dhash(img)
                is_dup = sha in seen_shas or any(
                    hex_hamming_distance(ah, pah) <= 8 or hex_hamming_distance(dh, pdh) <= 8
                    for _, pah, pdh in seen_phashes
                )
                if is_dup:
                    out_path.unlink(missing_ok=True)
                    continue

                rel = str(out_path.relative_to(base_dir)).replace("\\", "/")
                seen_shas.add(sha)
                seen_phashes.append((rel, ah, dh))
                seen_urls.add(hit["image_url"])
                seen_titles.add(hit["title"].lower())
                kept.append({
                    "id": f"{group}_cat_{idx:03d}",
                    "group": group,
                    "rel_path": rel,
                    **hit,
                    "width": iw,
                    "height": ih,
                    "sha256": sha,
                    "ahash": ah,
                    "dhash": dh,
                })
                print(f"  [{group}] {len(kept)}/{target}: {hit['title']} ({iw}x{ih}, {hit['license']})", flush=True)

        meta_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also build contact sheets for _staging_cats
    sheets_dir = base_dir / "manifests" / "contact_sheets_cats"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for group in OBJECT_IDS:
        gdir = staging_cats / group
        imgs = list_images(gdir)
        if not imgs:
            continue
        cols = 5
        cw, ch = 260, 220
        rows = (len(imgs) + cols - 1) // cols
        canvas = np.full((rows * ch, cols * cw, 3), 245, dtype=np.uint8)
        for i, p in enumerate(imgs):
            r, c = divmod(i, cols)
            x0, y0 = c * cw, r * ch
            img = load_image_unicode(p)
            ih, iw = img.shape[:2]
            sc = min((cw - 16) / iw, (ch - 38) / ih)
            nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
            th = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            tx = x0 + (cw - nw) // 2
            ty = y0 + 28 + ((ch - 34) - nh) // 2
            canvas[ty:ty + nh, tx:tx + nw] = th
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cw - 3, y0 + ch - 3), (200, 200, 200), 1)
            cv2.putText(canvas, p.stem, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        save_image_unicode(sheets_dir / f"contact_cat_{group}.jpg", canvas)


if __name__ == "__main__":
    download_category_supplements(Path(__file__).resolve().parent.parent)
