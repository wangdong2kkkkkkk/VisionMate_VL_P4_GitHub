"""Step 3: Collect verified open-licensed Wikimedia Commons photos for VL-P1b.

Collects fresh candidate images disjoint from all 125 VL-P1 images across:
- 5 target object categories (positives for calibration_v1b + holdout_v1b)
- 7 hard-negative confuser categories (for hard_neg_gallery + calibration_v1b + holdout_v1b)
- General indoor/outdoor distractor scenes (for holdout_v1b negatives)
- Ambiguous boundary candidates (for ambiguous_boundary split)
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import numpy as np
import requests

from .common_utils import (
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    list_images,
    load_image_unicode,
    make_contact_sheet,
    save_image_unicode,
)
from .labeling_protocol import is_allowed_open_license

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VisionMateVLP1bResearch/1.0 (https://github.com/visionmate-research; academic benchmark)"

EXCLUDE_TITLE_WORDS = (
    "logo", "icon", "diagram", "chart", "map", "flag", "coat of arms", "seal of",
    "poster", "advertisement", "newspaper", "cartoon", "drawing", "sketch", "painting",
    "illustration", "stamp", "coin", "medal", "blueprint", "patent", "svg", "score",
    "sheet music", "herbarium", "fossil", "mushroom", "butterfly", "beetle", "fish",
    "orchid", "candy", "sugar cane", "sugarcane", "bitcoin", "crypto", "screenshot",
    "florida keys", "keyboard", "piano", "typewriter", "organ", "saxophone", "flute",
    "horseshoe", "brake shoe", "snowshoe", "statue", "sculpture",
)

# Categories & queries for 5 positive target classes (need >= 25 new distinct per class: 5 cal + 20 holdout)
POSITIVE_SOURCES = {
    "object_01": {
        "categories": [
            "Category:White_canes",
            "Category:Blind_people_with_white_canes",
            "Category:International_White_Cane_Safety_Day",
        ],
        "queries": [
            '"white cane" blind -sugar -candy',
            'intitle:"white cane" -sugar -candy',
            '"long cane" blind mobility',
            'intitle:"Blindenstock" OR intitle:"blindenstok"',
            '"blind cane" walking OR street',
        ],
        "target": 45,
    },
    "object_02": {
        "categories": [
            "Category:Key_rings",
            "Category:Keys",
            "Category:House_keys",
            "Category:Car_keys",
        ],
        "queries": [
            'intitle:"bunch of keys" -florida -piano -map',
            'intitle:"keychain" keys -usb',
            'intitle:"keyring" OR intitle:"key ring" -chain',
            'intitle:"Schlüsselbund"',
            '"house keys" OR "car keys" ring',
        ],
        "target": 45,
    },
    "object_03": {
        "categories": [
            "Category:Wallets",
            "Category:Leather_wallets",
            "Category:Coin_purses",
            "Category:Bifold_wallets",
        ],
        "queries": [
            'intitle:"wallet" leather -bitcoin -crypto',
            'intitle:"coin purse" OR intitle:"money purse"',
            'intitle:"Portemonnaie" OR intitle:"Portefeuille" -castle',
            '"billfold" leather',
            '"zip wallet" leather',
        ],
        "target": 45,
    },
    "object_04": {
        "categories": [
            "Category:Flip-flops_(footwear)",
            "Category:Slippers",
            "Category:Sandals",
            "Category:Zori",
        ],
        "queries": [
            'intitle:"flip-flops" OR intitle:"flip flops"',
            'intitle:"slippers" -orchid -lobster -fish',
            'intitle:"Hausschuhe" OR intitle:"Pantoffel"',
            '"slide sandals" OR "rubber sandals"',
            'intitle:"žabky" OR intitle:"sandals" beach',
        ],
        "target": 45,
    },
    "object_05": {
        "categories": [
            "Category:Sneakers",
            "Category:Running_shoes",
            "Category:Dress_shoes",
            "Category:Oxford_shoes",
            "Category:Leather_shoes",
        ],
        "queries": [
            'intitle:"sneakers" -building',
            'intitle:"running shoes" OR intitle:"training shoes"',
            'intitle:"leather shoes" OR intitle:"dress shoes"',
            'intitle:"Turnschuhe" OR intitle:"Lederschuhe"',
            'intitle:"athletic shoes" OR intitle:"basketball shoes"',
        ],
        "target": 45,
    },
}

# 7 hard-negative groups + general household scenes
NEGATIVE_SOURCES = {
    "glasses_case_confuser": {
        "categories": ["Category:Spectacle_cases", "Category:Sunglasses"],
        "queries": ['intitle:"glasses case"', 'intitle:"spectacle case"', 'intitle:"eyeglass case"', '"Brillenetui"'],
        "target": 26,
    },
    "notebook_pouch_confuser": {
        "categories": ["Category:Notebooks", "Category:Leather_bindings", "Category:Pencil_cases"],
        "queries": ['intitle:"notebook" leather', 'intitle:"Moleskine"', 'intitle:"pouch" zipper', 'intitle:"Notizbuch"'],
        "target": 26,
    },
    "walking_stick_confuser": {
        "categories": ["Category:Walking_sticks", "Category:Trekking_poles", "Category:Crutches"],
        "queries": ['intitle:"walking stick" wooden -insect -phasmid', 'intitle:"trekking poles"', 'intitle:"Spazierstock" OR intitle:"Wanderstock"'],
        "target": 26,
    },
    "long_metal_rod_confuser": {
        "categories": ["Category:Selfie_sticks", "Category:Ski_poles", "Category:Golf_clubs"],
        "queries": ['intitle:"selfie stick"', 'intitle:"ski poles"', 'intitle:"crowbar" OR intitle:"metal rod" tool'],
        "target": 20,
    },
    "lock_key_metal_confuser": {
        "categories": ["Category:Padlocks", "Category:Scissors", "Category:Spoons", "Category:Carabiners"],
        "queries": ['intitle:"padlock" brass -key', 'intitle:"scissors" metal', 'intitle:"spoons" cutlery', 'intitle:"carabiner"'],
        "target": 26,
    },
    "footwear_sock_confuser": {
        "categories": ["Category:Socks", "Category:Geta", "Category:Roller_skates"],
        "queries": ['intitle:"socks" cotton OR wool', 'intitle:"Socken"', 'intitle:"ice skates"', 'intitle:"shoe insoles"'],
        "target": 20,
    },
    "umbrella_handle_confuser": {
        "categories": ["Category:Umbrellas", "Category:Folded_umbrellas"],
        "queries": ['intitle:"umbrella" handle OR folded', 'intitle:"Regenschirm"'],
        "target": 20,
    },
    "general_distractor": {
        "categories": [
            "Category:Wristwatches",
            "Category:Computer_mice",
            "Category:Coffee_mugs",
            "Category:Water_bottles",
            "Category:Flashlights",
            "Category:Eyeglasses",
            "Category:Mobile_phones",
            "Category:Backpacks",
        ],
        "queries": [
            'intitle:"wristwatch" table',
            'intitle:"computer mouse"',
            'intitle:"flashlight" OR intitle:"torch"',
            'intitle:"water bottle"',
            'intitle:"tv remote" OR intitle:"remote control"',
        ],
        "target": 55,
    },
}

# Ambiguous boundary cases to report separately
AMBIGUOUS_SOURCES = {
    "ambiguous_boundary": {
        "categories": ["Category:Clogs", "Category:Antique_keys"],
        "queries": [
            'intitle:"single key" brass OR iron',
            'intitle:"sabots" OR intitle:"wooden clogs"',
            'intitle:"coin pouch" cloth',
        ],
        "target": 12,
    }
}


def _strip_html(raw: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", raw or "")
    return " ".join(txt.split())


def _normalize_title(title: str) -> str:
    t = title.strip().lower()
    if t.startswith("file:"):
        t = t[5:]
    return re.sub(r"[\s_]+", "_", t)


def _is_clean_title(title: str) -> bool:
    tl = title.lower()
    if not tl.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    return not any(bad in tl for bad in EXCLUDE_TITLE_WORDS)


def _load_vl_p1_signatures(vl_p1_dir: Path) -> tuple[set[str], set[str], list[tuple[str, str, str]]]:
    """Load all SHA-256s, Wikimedia titles/URLs, and pHashes of all 125 VL-P1 images + raw data."""
    seen_shas: set[str] = set()
    seen_titles: set[str] = set()
    seen_phashes: list[tuple[str, str, str]] = []

    manifest_p = vl_p1_dir / "manifests" / "dataset_manifest.json"
    if manifest_p.is_file():
        mf = json.loads(manifest_p.read_text(encoding="utf-8"))
        for split_key in ("train", "calibration", "test"):
            for item in mf.get(split_key, []):
                if item.get("sha256"):
                    seen_shas.add(item["sha256"])
                if item.get("full_sha256"):
                    seen_shas.add(item["full_sha256"])
                if item.get("title"):
                    seen_titles.add(_normalize_title(item["title"]))
                if item.get("source_url"):
                    last = item["source_url"].rstrip("/").split("/")[-1]
                    seen_titles.add(_normalize_title(last))
                if item.get("ahash") and item.get("dhash"):
                    seen_phashes.append((item["id"], item["ahash"], item["dhash"]))
                if item.get("full_ahash") and item.get("full_dhash"):
                    seen_phashes.append((item["id"] + "_full", item["full_ahash"], item["full_dhash"]))

    # Also scan every image file physically under vl_p1/data
    data_root = vl_p1_dir / "data"
    if data_root.exists():
        for p in data_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                if "_staging" in str(p):
                    continue
                seen_shas.add(compute_sha256(p))
                try:
                    img = load_image_unicode(p)
                    seen_phashes.append((p.name, compute_ahash(img), compute_dhash(img)))
                except ValueError:
                    pass
    return seen_shas, seen_titles, seen_phashes


def _api_fetch(session: requests.Session, params: dict) -> list[dict]:
    base_params = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 800,
        "format": "json",
    }
    base_params.update(params)
    for attempt in range(3):
        try:
            r = session.get(COMMONS_API, params=base_params, timeout=25)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            out = []
            for _, page in sorted(pages.items(), key=lambda kv: int(kv[0])):
                title = page.get("title", "")
                if not _is_clean_title(title):
                    continue
                iis = page.get("imageinfo", [])
                if not iis:
                    continue
                ii = iis[0]
                if ii.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
                    continue
                w = int(ii.get("thumbwidth") or ii.get("width") or 0)
                h = int(ii.get("thumbheight") or ii.get("height") or 0)
                if min(w, h) < 160 or max(w, h) > 3000:
                    continue
                meta = ii.get("extmetadata", {})
                lic = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
                if not is_allowed_open_license(lic):
                    continue
                out.append({
                    "title": title,
                    "norm_title": _normalize_title(title),
                    "page_url": ii.get("descriptionurl", ""),
                    "image_url": ii.get("thumburl") or ii.get("url"),
                    "orig_url": ii.get("url", ""),
                    "license": lic,
                    "author": _strip_html(meta.get("Artist", {}).get("value", "Unknown"))[:120],
                    "description": _strip_html(meta.get("ImageDescription", {}).get("value", ""))[:220],
                })
            return out
        except Exception:
            time.sleep(1.0)
    return []


def collect_staging_pool(vl_p1_dir: Path, vl_p1b_dir: Path, sleep_sec: float = 0.25) -> dict:
    staging_root = vl_p1b_dir / "data" / "_staging_pool"
    staging_root.mkdir(parents=True, exist_ok=True)
    meta_file = staging_root / "staging_metadata.json"

    seen_shas, seen_titles, seen_phashes = _load_vl_p1_signatures(vl_p1_dir)
    existing_meta: dict[str, list[dict]] = {}
    if meta_file.is_file():
        existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for lst in existing_meta.values():
            for rec in lst:
                seen_shas.add(rec["sha256"])
                seen_titles.add(rec["norm_title"])
                seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_specs = {}
    all_specs.update({f"pos/{k}": v for k, v in POSITIVE_SOURCES.items()})
    all_specs.update({f"neg/{k}": v for k, v in NEGATIVE_SOURCES.items()})
    all_specs.update({f"amb/{k}": v for k, v in AMBIGUOUS_SOURCES.items()})

    for group_key, spec in all_specs.items():
        group_dir = staging_root / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        recs = existing_meta.get(group_key, [])
        recs = [r for r in recs if (staging_root / r["local_rel"]).is_file()]
        target = spec["target"]
        if len(recs) >= target:
            existing_meta[group_key] = recs
            print(f"[Skip] {group_key}: already has {len(recs)}/{target}", flush=True)
            continue

        candidates: list[dict] = []
        seen_batch_titles: set[str] = set()
        for cat in spec.get("categories", []):
            if len(candidates) >= target * 3:
                break
            rows = _api_fetch(session, {
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": 40,
            })
            for row in rows:
                if row["norm_title"] not in seen_batch_titles:
                    seen_batch_titles.add(row["norm_title"])
                    candidates.append(row)
            time.sleep(sleep_sec)

        if len(candidates) < target * 2:
            for q in spec.get("queries", []):
                if len(candidates) >= target * 3:
                    break
                rows = _api_fetch(session, {
                    "generator": "search",
                    "gsrsearch": f"{q} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 30,
                })
                for row in rows:
                    if row["norm_title"] not in seen_batch_titles:
                        seen_batch_titles.add(row["norm_title"])
                        candidates.append(row)
                time.sleep(sleep_sec)

        idx_counter = len(recs) + 1
        for cand in candidates:
            if len(recs) >= target:
                break
            if cand["norm_title"] in seen_titles:
                continue
            url = cand["image_url"]
            if not url:
                continue
            try:
                resp = session.get(url, timeout=25)
                if resp.status_code != 200 or len(resp.content) < 4096:
                    time.sleep(sleep_sec)
                    continue
                sha = re.sub(r"[^0-9a-f]", "", __import__("hashlib").sha256(resp.content).hexdigest())
                if sha in seen_shas:
                    continue
                import cv2
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None or min(img.shape[:2]) < 150:
                    continue
                ah = compute_ahash(img)
                dh = compute_dhash(img)
                is_dup = False
                for _, old_ah, old_dh in seen_phashes:
                    if min(hex_hamming_distance(ah, old_ah), hex_hamming_distance(dh, old_dh)) <= 6:
                        is_dup = True
                        break
                if is_dup:
                    continue

                fname = f"cand_{idx_counter:03d}.jpg"
                out_p = group_dir / fname
                save_image_unicode(out_p, img)
                rel_p = out_p.relative_to(staging_root).as_posix()
                rec = {
                    "local_rel": rel_p,
                    "group_key": group_key,
                    "title": cand["title"],
                    "norm_title": cand["norm_title"],
                    "source_url": cand["page_url"],
                    "image_url": url,
                    "license": cand["license"],
                    "author": cand["author"],
                    "description": cand["description"],
                    "sha256": compute_sha256(out_p),
                    "ahash": ah,
                    "dhash": dh,
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "session_key": cand["norm_title"].split("_(")[0][:40],
                }
                recs.append(rec)
                seen_shas.add(rec["sha256"])
                seen_shas.add(sha)
                seen_titles.add(cand["norm_title"])
                seen_phashes.append((rel_p, ah, dh))
                idx_counter += 1
                time.sleep(sleep_sec)
            except Exception:
                time.sleep(sleep_sec)

        existing_meta[group_key] = recs
        meta_file.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Collected] {group_key}: {len(recs)}/{target}", flush=True)

        # Build visual contact sheet for auditing candidates
        cs_items = []
        for r in recs:
            p = staging_root / r["local_rel"]
            try:
                cs_items.append((f"{p.stem} | {r['title'][5:]}", load_image_unicode(p), None))
            except ValueError:
                pass
        if cs_items:
            sheet_name = group_key.replace("/", "_") + ".jpg"
            make_contact_sheet(cs_items, vl_p1b_dir / "manifests" / "contact_sheets_v1b" / sheet_name, cols=5)

    return existing_meta


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    collect_staging_pool(root / "research" / "vl_p1", root / "research" / "vl_p1b")
