"""Collect and verify open-licensed Wikimedia Commons images for VL-P1 calibration and test splits.

Includes strict anti-leakage checks against existing train/originals/dev_probes,
metadata logging (URL, author, license, SHA-256, dHash, aHash), and contact-sheet
generation for visual quality verification.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests

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

USER_AGENT = "VisionMateVLP1Research/1.0 (https://github.com/visionmate-research; educational research)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

QUERY_SPECS = {
    "object_01": {
        "queries": [
            '"white cane" blind',
            'intitle:"white cane"',
            '"mobility cane" blind',
            'Blindenschirm OR Blindenstock',
        ],
        "exclude_terms": ["sugar", "candy", "bamboo", "rattan", "fishing", "insect", "flower", "mushroom", "chart", "map", "logo", "icon"],
        "target_count": 20,
    },
    "object_02": {
        "queries": [
            'intitle:"bunch of keys"',
            'intitle:"keyring" keys',
            '"house keys" keyring',
            '"keychain" metal keys',
        ],
        "exclude_terms": ["piano", "keyboard", "typewriter", "map", "clef", "music", "chart", "diagram", "florida", "usb", "car key fob only"],
        "target_count": 20,
    },
    "object_03": {
        "queries": [
            '"leather wallet"',
            'intitle:"wallet" leather',
            'intitle:"bifold wallet" OR intitle:"billfold"',
            '"coin purse" wallet',
        ],
        "exclude_terms": ["bitcoin", "crypto", "ethereum", "app", "software", "screenshot", "logo", "icon", "diagram", "monero"],
        "target_count": 20,
    },
    "object_04": {
        "queries": [
            'intitle:"flip-flops" OR intitle:"flip flops"',
            'intitle:"slippers" footwear',
            '"rubber slippers" OR "slide sandals"',
            'intitle:"sandals" pair',
        ],
        "exclude_terms": ["orchid", "lobster", "fish", "limpet", "flower", "butterfly", "logo", "diagram", "statue", "fossil"],
        "target_count": 20,
    },
    "object_05": {
        "queries": [
            'intitle:"sneakers" pair',
            'intitle:"running shoes" pair',
            'intitle:"leather shoes" pair',
            '"dress shoes" leather pair',
        ],
        "exclude_terms": ["horseshoe", "brake shoe", "snowshoe", "building", "logo", "icon", "diagram", "print", "footprint"],
        "target_count": 20,
    },
    "unknown": {
        "queries": [
            'intitle:"walking stick" wooden',
            'intitle:"trekking poles" OR intitle:"hiking pole"',
            'intitle:"folded umbrella"',
            'intitle:"eyeglasses case" OR intitle:"spectacle case"',
            'intitle:"padlock" brass',
            'intitle:"scissors" metal',
            'intitle:"spoon" cutlery',
            'intitle:"notebook" leather',
            'intitle:"coin" macro',
            'intitle:"socks" pair',
            'intitle:"coffee mug" table',
            'intitle:"wrench" spanner',
        ],
        "exclude_terms": ["white cane", "blind", "wallet", "billfold", "keychain", "keyring", "slipper", "flip-flop", "sneaker", "logo", "diagram"],
        "target_count": 50,
    },
}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _existing_signatures(base_dir: Path) -> tuple[set[str], list[tuple[str, str, str]], set[str]]:
    """Collect SHA-256, perceptual hashes, and Wikimedia filenames from existing train/originals/probes."""
    shas: set[str] = set()
    phashes: list[tuple[str, str, str]] = []
    known_titles: set[str] = set()

    for rel in ("data/originals", "data/train", "dev_probes", "backup_crops"):
        root = base_dir / rel
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                shas.add(compute_sha256(p))
                try:
                    img = load_image_unicode(p)
                    phashes.append((str(p.relative_to(base_dir)), compute_ahash(img), compute_dhash(img)))
                except ValueError:
                    pass

    ps_path = base_dir / "photo_sources.json"
    if ps_path.exists():
        text = ps_path.read_text(encoding="utf-8")
        for m in re.findall(r"File:([^\s\"']+)", text):
            known_titles.add(m.lower().replace("_", " "))
    csv_path = base_dir / "downloaded_photo_sources.csv"
    if csv_path.exists():
        text = csv_path.read_text(encoding="utf-8-sig")
        for m in re.findall(r"File:([^\s\",]+)", text):
            known_titles.add(m.lower().replace("_", " "))

    return shas, phashes, known_titles


def search_wikimedia(query: str, limit: int = 25) -> list[dict]:
    """Query Wikimedia Commons API for image files and metadata."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,  # File:
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 800,
        "format": "json",
    }
    resp = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    results = []
    for _, page in sorted(pages.items(), key=lambda kv: int(kv[0])):
        title = page.get("title", "")
        iis = page.get("imageinfo", [])
        if not iis:
            continue
        ii = iis[0]
        mime = ii.get("mime", "")
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        meta = ii.get("extmetadata", {})
        license_name = _strip_html(meta.get("LicenseShortName", {}).get("value", "Unknown"))
        if not any(k in license_name.upper() for k in ("CC", "PUBLIC DOMAIN", "CC0", "PDM")):
            continue
        author = _strip_html(meta.get("Artist", {}).get("value", "Unknown"))
        desc = _strip_html(meta.get("ImageDescription", {}).get("value", ""))
        thumb_url = ii.get("thumburl") or ii.get("url")
        results.append({
            "title": title,
            "page_url": ii.get("descriptionurl", ""),
            "image_url": thumb_url,
            "orig_url": ii.get("url", ""),
            "width": ii.get("thumbwidth") or ii.get("width", 0),
            "height": ii.get("thumbheight") or ii.get("height", 0),
            "license": license_name,
            "author": author[:120],
            "description": desc[:250],
        })
    return results


def download_candidates(base_dir: Path, sleep_sec: float = 0.35) -> dict:
    """Download candidate images into data/_staging/<group>/ with strict dedup and metadata logging."""
    staging_dir = base_dir / "data" / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    seen_shas, seen_phashes, banned_titles = _existing_signatures(base_dir)

    candidates_log: dict[str, list[dict]] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for group, spec in QUERY_SPECS.items():
        group_dir = staging_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        existing_meta_path = group_dir / "metadata.json"
        if existing_meta_path.exists():
            kept = json.loads(existing_meta_path.read_text(encoding="utf-8"))
            for item in kept:
                p = base_dir / item["rel_path"]
                if p.exists():
                    seen_shas.add(item["sha256"])
                    seen_phashes.append((item["rel_path"], item["ahash"], item["dhash"]))
        else:
            kept = []

        target = spec["target_count"]
        excludes = [t.lower() for t in spec["exclude_terms"]]
        seen_urls = {k["image_url"] for k in kept}
        seen_titles = {k["title"].lower() for k in kept}

        for q in spec["queries"]:
            if len(kept) >= target:
                break
            try:
                hits = search_wikimedia(q, limit=25)
            except Exception as exc:
                print(f"[WARN] Search failed for {group} / {q}: {exc}")
                continue
            for hit in hits:
                if len(kept) >= target:
                    break
                title_clean = hit["title"].replace("File:", "").lower().replace("_", " ")
                if any(b in title_clean for b in banned_titles):
                    continue
                haystack = f"{title_clean} {hit['description'].lower()}"
                if any(ex in haystack for ex in excludes):
                    continue
                if hit["image_url"] in seen_urls or hit["title"].lower() in seen_titles:
                    continue
                if min(hit["width"], hit["height"]) < 180:
                    continue

                time.sleep(sleep_sec)
                try:
                    r = session.get(hit["image_url"], timeout=25)
                    r.raise_for_status()
                except Exception as exc:
                    continue
                raw = r.content
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue
                ih, iw = img.shape[:2]
                if min(ih, iw) < 160 or max(ih / max(1, iw), iw / max(1, ih)) > 4.5:
                    continue

                idx = len(kept) + 1
                out_path = group_dir / f"cand_{idx:03d}.jpg"
                save_image_unicode(out_path, img)
                sha = compute_sha256(out_path)
                ah = compute_ahash(img)
                dh = compute_dhash(img)

                is_dup = sha in seen_shas
                if not is_dup:
                    for _, prev_ah, prev_dh in seen_phashes:
                        if hex_hamming_distance(ah, prev_ah) <= 8 or hex_hamming_distance(dh, prev_dh) <= 8:
                            is_dup = True
                            break
                if is_dup:
                    out_path.unlink(missing_ok=True)
                    continue

                rel = str(out_path.relative_to(base_dir)).replace("\\", "/")
                seen_shas.add(sha)
                seen_phashes.append((rel, ah, dh))
                seen_urls.add(hit["image_url"])
                seen_titles.add(hit["title"].lower())

                kept.append({
                    "id": f"{group}_cand_{idx:03d}",
                    "group": group,
                    "rel_path": rel,
                    "title": hit["title"],
                    "page_url": hit["page_url"],
                    "image_url": hit["image_url"],
                    "orig_url": hit["orig_url"],
                    "author": hit["author"],
                    "license": hit["license"],
                    "description": hit["description"],
                    "width": iw,
                    "height": ih,
                    "sha256": sha,
                    "ahash": ah,
                    "dhash": dh,
                    "query": q,
                })
                print(f"  [{group}] {len(kept)}/{target}: {hit['title']} ({iw}x{ih}, {hit['license']})")

        existing_meta_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
        candidates_log[group] = kept
    return candidates_log


def build_contact_sheets(base_dir: Path, out_dir: Path) -> list[Path]:
    """Create labeled thumbnail contact sheets for every staging group for visual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = base_dir / "data" / "_staging"
    sheets = []
    for group in [*OBJECT_IDS, "unknown"]:
        gdir = staging / group
        imgs = list_images(gdir)
        if not imgs:
            continue
        cols = 5
        cell_w, cell_h = 260, 220
        rows = (len(imgs) + cols - 1) // cols
        canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
        for i, p in enumerate(imgs):
            r, c = divmod(i, cols)
            x0, y0 = c * cell_w, r * cell_h
            img = load_image_unicode(p)
            ih, iw = img.shape[:2]
            scale = min((cell_w - 16) / iw, (cell_h - 38) / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            tx = x0 + (cell_w - nw) // 2
            ty = y0 + 28 + ((cell_h - 34) - nh) // 2
            canvas[ty:ty + nh, tx:tx + nw] = thumb
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cell_w - 3, y0 + cell_h - 3), (200, 200, 200), 1)
            cv2.putText(canvas, p.stem, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        sheet_path = out_dir / f"contact_{group}.jpg"
        save_image_unicode(sheet_path, canvas)
        sheets.append(sheet_path)
    return sheets


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    download_candidates(base)
    sheets_dir = base / "manifests" / "contact_sheets"
    sheets = build_contact_sheets(base, sheets_dir)
    print("Generated contact sheets:", [str(s) for s in sheets])
