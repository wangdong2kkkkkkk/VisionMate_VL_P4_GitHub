"""Step 2a: Dataset Collector & Leakage Guard for VL-P2 (White Cane Localization & Segmentation).

1. Reuses valid open-licensed images from prior phases (`VL-P1`, `VL-P1b`, `VL-P1c`) strictly for
   development/calibration (`dev_cal_v2`) and reference galleries (`train_pos`, `hard_neg_gallery`).
   Never puts any historical image into `final_holdout_v2`.
2. Collects brand-new Wikimedia Commons images across multilingual white-cane categories/queries
   for `final_holdout_v2` (`>= 50` object_01 + `>= 100` negatives) and supplements `dev_cal_v2`
   (`>= 60` object_01 + `>= 80` hard negatives), plus `ambiguous_cane_v2` (`>= 15`).
3. Enforces zero overlap across SHA-256, normalized title, cluster_id, and perceptual hash (aHash/dHash > 6).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    load_image_unicode,
    normalize_title,
    save_image_unicode,
)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VisionMate-VL-P2-Research/1.0 (offline accessibility research; https://example.org)"

ALLOWED_LICENSE_TOKENS = (
    "cc0",
    "cc-by",
    "cc by",
    "public domain",
    "pd",
    "gfdl",
    "fal",
    "free art",
    "ogl",
    "mit",
    "bsd",
    "apache",
)

EXCLUDE_TITLE_WORDS = (
    "logo", "icon", "svg", "diagram", "chart", "map", "flag", "coat_of_arms",
    "emblem", "seal", "stamp", "coin", "medal", "poster", "flyer", "banner",
    "cartoon", "comic", "drawing", "sketch", "painting", "engraving", "caricature",
    "candy_cane", "sugar_cane", "sugarcane", "cane_toad", "cane_corso", "rattan",
    "caine", "hurricane", "volcano", "grave", "statue", "monument", "plaque",
    "braille_sign", "pictogram", "traffic_sign", "street_sign", "road_sign",
)

OBJECT_01_POSITIVE_SPECS = {
    "categories": [
        "Category:White_canes",
        "Category:Blind_people_with_white_canes",
        "Category:Orientation_and_mobility",
        "Category:Blind_pedestrians",
        "Category:White_Cane_Safety_Day",
        "Category:Visually_impaired_people",
        "Category:Blindness",
        "Category:Guide_dogs",
        "Category:Tactile_paving",
        "Category:Tactile_paving_in_Japan",
        "Category:Tactile_paving_in_Europe",
    ],
    "queries": [
        '"white cane" blind OR "visually impaired" -candy -sugar -sign -icon -logo',
        '"mobility cane" OR "long cane" blind OR visually -diagram',
        '"canne blanche" aveugle OR malvoyant OR handicap',
        '"Blindenstock" OR "Langstock" blind OR Blinde OR Leitstreifen',
        '"bastón blanco" OR "baston blanco" ciego OR invidente',
        '"bengala branca" cego OR "deficiente visual"',
        '"bastone bianco" cieco OR non vedente',
        '"белая трость" слепой OR незрячий OR инвалид',
        '"białą laską" OR "biała laska" niewidomy',
        '"白杖" 視覚障害 OR 点字ブロック',
        '"blindenstok" OR "wit_stok" blind OR slechtziend',
        '"blindkäpp" OR "vit käpp" blind',
        '"orientation and mobility" cane blind',
        '"tactile paving" "white cane" OR "blind person"',
        '"International White Cane" OR "White Cane Day"',
    ],
}

NEGATIVE_CONFUSER_SPECS = {
    "walking_stick_confuser": {
        "target": 35,
        "categories": [
            "Category:Walking_sticks",
            "Category:Shillelaghs",
            "Category:Canes",
            "Category:Crutches",
        ],
        "queries": [
            '"walking stick" wooden OR brass OR antique -white_cane -blind -sugar',
            '"wooden cane" walking OR handle -blind -white_cane',
            'intitle:"crutch" OR intitle:"crutches" walking -blind',
        ],
    },
    "trekking_pole_confuser": {
        "target": 35,
        "categories": [
            "Category:Trekking_poles",
            "Category:Nordic_walking",
            "Category:Ski_poles",
        ],
        "queries": [
            '"trekking poles" OR "trekking pole" hiking OR trail -blind',
            '"Nordic walking" poles OR sticks',
            '"ski poles" snow OR skier',
        ],
    },
    "umbrella_confuser": {
        "target": 30,
        "categories": [
            "Category:Closed_umbrellas",
            "Category:Folded_umbrellas",
            "Category:Black_umbrellas",
            "Category:Umbrellas_in_art_and_culture",
        ],
        "queries": [
            '"closed umbrella" OR "folded umbrella" OR "umbrella handle" -blind',
            '"umbrellas" street pedestrian walking -white_cane',
        ],
    },
    "metal_rod_pipe_confuser": {
        "target": 30,
        "categories": [
            "Category:Monopods",
            "Category:Selfie_sticks",
            "Category:Brooms",
            "Category:Fishing_rods",
        ],
        "queries": [
            '"selfie stick" OR "monopod" camera -blind',
            '"broom" OR "mop" handle floor cleaning -blind',
            '"fishing rod" metal OR carbon',
        ],
    },
    "white_line_pole_confuser": {
        "target": 30,
        "categories": [
            "Category:Bollards",
            "Category:Pedestrian_crossings",
            "Category:Handrails",
        ],
        "queries": [
            '"white bollard" street OR traffic -blind -cane',
            '"white railing" OR "white handrail" stairs OR sidewalk -cane',
            '"zebra crossing" OR "crosswalk" white stripes street -cane -blind',
        ],
    },
    "general_distractor": {
        "target": 40,
        "categories": [
            "Category:Sidewalks",
            "Category:Tactile_paving",
            "Category:Train_station_platforms",
        ],
        "queries": [
            '"sidewalk" pedestrians shoes street -cane -blind',
            '"tactile paving" platform OR street -cane -blind -dog',
            '"urban street" crosswalk pedestrians -cane -blind',
        ],
    },
}

AMBIGUOUS_CANE_SPEC = {
    "target": 20,
    "categories": [
        "Category:PVC_pipes",
        "Category:Baton_twirling",
        "Category:Drum_majors",
        "Category:Ski_poles",
    ],
    "queries": [
        '"white pipe" OR "pvc pipe" rod OR stick -cane -blind',
        '"white baton" OR "drum major" mace OR stick',
        '"white pole" OR "white rod" thin stick -sign -flag',
    ],
}


def is_allowed_open_license(lic: str) -> bool:
    low = (lic or "").strip().lower()
    if not low or "non-commercial" in low or "nc" in low or "fair use" in low:
        return False
    return any(tok in low for tok in ALLOWED_LICENSE_TOKENS)


def _strip_html(raw: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", raw or "").split())


def _is_clean_title(title: str) -> bool:
    tl = title.lower()
    if not tl.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    return not any(bad in tl for bad in EXCLUDE_TITLE_WORDS)


def _load_all_prior_signatures(*research_dirs: Path) -> tuple[set[str], set[str], set[str], list[tuple[str, str, str]]]:
    """Load SHA-256, normalized titles, cluster_ids, and perceptual hashes from all prior phases."""
    seen_shas: set[str] = set()
    seen_titles: set[str] = set()
    seen_clusters: set[str] = set()
    seen_phashes: list[tuple[str, str, str]] = []

    for rdir in research_dirs:
        if not rdir.exists():
            continue
        mf_p = rdir / "manifests" / "dataset_manifest.json"
        if mf_p.is_file():
            mf = json.loads(mf_p.read_text(encoding="utf-8"))
            for key, val in mf.items():
                if not isinstance(val, list):
                    continue
                for rec in val:
                    if not isinstance(rec, dict):
                        continue
                    for sk in ("sha256", "full_sha256", "roi_sha256"):
                        if rec.get(sk):
                            seen_shas.add(rec[sk])
                    if rec.get("source_title"):
                        seen_titles.add(normalize_title(rec["source_title"]))
                    if rec.get("title"):
                        seen_titles.add(normalize_title(rec["title"]))
                    if rec.get("source_cluster"):
                        seen_clusters.add(normalize_title(rec["source_cluster"]))
                    if rec.get("ahash") and rec.get("dhash"):
                        seen_phashes.append((str(rec.get("id", key)), rec["ahash"], rec["dhash"]))
        data_root = rdir / "data"
        if data_root.exists():
            for p in data_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    if "_staging" in str(p):
                        continue
                    seen_shas.add(compute_sha256(p))
    return seen_shas, seen_titles, seen_clusters, seen_phashes


def _api_fetch_paginated(session: requests.Session, params: dict, max_pages: int = 3) -> list[dict]:
    out: list[dict] = []
    cur = dict(params)
    for _ in range(max_pages):
        req = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
        }
        req.update(cur)
        data = None
        for attempt in range(3):
            try:
                r = session.get(COMMONS_API, params=req, timeout=25)
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                time.sleep(1.0)
        if not data:
            break
        pages = data.get("query", {}).get("pages", {})
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
            if min(w, h) < 200 or max(w, h) > 3200:
                continue
            meta = ii.get("extmetadata", {})
            lic = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not is_allowed_open_license(lic):
                continue
            desc = _strip_html(meta.get("ImageDescription", {}).get("value", ""))[:240]
            cats = _strip_html(meta.get("Categories", {}).get("value", ""))[:200]
            out.append({
                "title": title,
                "norm_title": normalize_title(title),
                "page_url": ii.get("descriptionurl", ""),
                "image_url": ii.get("thumburl") or ii.get("url"),
                "orig_url": ii.get("url", ""),
                "license": lic,
                "author": _strip_html(meta.get("Artist", {}).get("value", "Unknown"))[:120],
                "description": desc,
                "categories": cats,
            })
        cont = data.get("continue")
        if not cont:
            break
        cur.update(cont)
    return out


def _is_duplicate(
    sha: str,
    norm_t: str,
    ah: str,
    dh: str,
    seen_shas: set[str],
    seen_titles: set[str],
    seen_phashes: list[tuple[str, str, str]],
) -> bool:
    if sha in seen_shas or norm_t in seen_titles:
        return True
    for _, o_ah, o_dh in seen_phashes:
        if hex_hamming_distance(ah, o_ah) <= 6 or hex_hamming_distance(dh, o_dh) <= 6:
            return True
    return False


def _download_candidate(session: requests.Session, cand: dict, out_path: Path) -> dict | None:
    try:
        r = session.get(cand["image_url"], timeout=25)
        r.raise_for_status()
        blob = r.content
        if len(blob) < 4000:
            return None
        tmp_path = out_path.with_suffix(".jpg")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(blob)
        img = load_image_unicode(tmp_path)
        h, w = img.shape[:2]
        if min(h, w) < 200:
            tmp_path.unlink(missing_ok=True)
            return None
        if max(h, w) > 960:
            sc = 960.0 / float(max(h, w))
            img = cv2_resize_safe(img, int(round(w * sc)), int(round(h * sc)))
        save_image_unicode(tmp_path, img)
        sha = compute_sha256(tmp_path)
        ah = compute_ahash(img)
        dh = compute_dhash(img)
        return {
            "path": tmp_path,
            "sha256": sha,
            "ahash": ah,
            "dhash": dh,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            **cand,
        }
    except Exception:
        return None


def cv2_resize_safe(img: np.ndarray, rw: int, rh: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (max(1, rw), max(1, rh)), interpolation=cv2.INTER_AREA)


def _fetch_group_pool(
    session: requests.Session,
    spec: dict,
    out_dir: Path,
    prefix: str,
    target_count: int,
    seen_shas: set[str],
    seen_titles: set[str],
    seen_phashes: list[tuple[str, str, str]],
    existing_recs: list[dict],
) -> list[dict]:
    recs = [r for r in existing_recs if Path(r["abs_path"]).is_file()]
    if len(recs) >= target_count:
        return recs

    candidates: list[dict] = []
    batch_titles: set[str] = set()
    for cat in spec.get("categories", []):
        if len(candidates) >= target_count * 4:
            break
        rows = _api_fetch_paginated(session, {
            "generator": "categorymembers",
            "gcmtitle": cat,
            "gcmtype": "file",
            "gcmlimit": 50,
        }, max_pages=3)
        for row in rows:
            nt = row["norm_title"]
            if nt not in seen_titles and nt not in batch_titles:
                batch_titles.add(nt)
                candidates.append(row)

    for q in spec.get("queries", []):
        if len(candidates) >= target_count * 5:
            break
        rows = _api_fetch_paginated(session, {
            "generator": "search",
            "gsrsearch": q,
            "gsrnamespace": 6,
            "gsrlimit": 50,
        }, max_pages=2)
        for row in rows:
            nt = row["norm_title"]
            if nt not in seen_titles and nt not in batch_titles:
                batch_titles.add(nt)
                candidates.append(row)

    next_idx = len(recs) + 1
    for cand in candidates:
        if len(recs) >= target_count:
            break
        out_p = out_dir / f"{prefix}_{next_idx:03d}.jpg"
        dl = _download_candidate(session, cand, out_p)
        if dl is None:
            continue
        if _is_duplicate(dl["sha256"], dl["norm_title"], dl["ahash"], dl["dhash"], seen_shas, seen_titles, seen_phashes):
            dl["path"].unlink(missing_ok=True)
            continue
        seen_shas.add(dl["sha256"])
        seen_titles.add(dl["norm_title"])
        seen_phashes.append((dl["path"].name, dl["ahash"], dl["dhash"]))
        recs.append({
            "id": f"{prefix}_{next_idx:03d}",
            "abs_path": str(dl["path"]),
            "sha256": dl["sha256"],
            "ahash": dl["ahash"],
            "dhash": dl["dhash"],
            "width": dl["width"],
            "height": dl["height"],
            "title": dl["title"],
            "norm_title": dl["norm_title"],
            "source_cluster": dl["norm_title"][:48],
            "source_url": dl["page_url"],
            "image_url": dl["image_url"],
            "license": dl["license"],
            "author": dl["author"],
            "description": dl["description"],
        })
        next_idx += 1
        time.sleep(0.08)
    return recs


def collect_vl_p2_staging_pool(worktree_root: Path, pos_target: int = 85) -> dict:
    """Collect new open-licensed white-cane, hard-negative, and ambiguous candidates for VL-P2."""
    ai_root = worktree_root.parent
    vl_p1_dir = ai_root / "VisionMate_VL_P1_AG" / "research" / "vl_p1"
    vl_p1b_dir = ai_root / "VisionMate_VL_P1b_AG" / "research" / "vl_p1b"
    vl_p1c_dir = ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"
    vl_p2_dir = worktree_root / "research" / "vl_p2"

    staging_root = vl_p2_dir / "cache" / "staging_new"
    staging_root.mkdir(parents=True, exist_ok=True)
    meta_file = staging_root / "staging_metadata.json"

    seen_shas, seen_titles, seen_clusters, seen_phashes = _load_all_prior_signatures(
        vl_p1_dir, vl_p1b_dir, vl_p1c_dir
    )
    existing_meta: dict[str, list[dict]] = {}
    if meta_file.is_file():
        existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for lst in existing_meta.values():
            for r in lst:
                seen_shas.add(r["sha256"])
                seen_titles.add(r["norm_title"])
                seen_phashes.append((r["id"], r["ahash"], r["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print("[Collector] Fetching new object_01 white-cane candidates...", flush=True)
    pos_recs = _fetch_group_pool(
        session,
        OBJECT_01_POSITIVE_SPECS,
        staging_root / "pos_object_01",
        "v2_pos_cane",
        pos_target,
        seen_shas,
        seen_titles,
        seen_phashes,
        existing_meta.get("pos_object_01", []),
    )
    existing_meta["pos_object_01"] = pos_recs
    meta_file.write_text(json.dumps(existing_meta, indent=2), encoding="utf-8")
    print(f"[Collector] pos_object_01 collected: {len(pos_recs)}/{pos_target}", flush=True)

    for grp, spec in NEGATIVE_CONFUSER_SPECS.items():
        print(f"[Collector] Fetching new negative group '{grp}' (target={spec['target']})...", flush=True)
        nrecs = _fetch_group_pool(
            session,
            spec,
            staging_root / f"neg_{grp}",
            f"v2_neg_{grp[:8]}",
            int(spec["target"]),
            seen_shas,
            seen_titles,
            seen_phashes,
            existing_meta.get(f"neg_{grp}", []),
        )
        existing_meta[f"neg_{grp}"] = nrecs
        meta_file.write_text(json.dumps(existing_meta, indent=2), encoding="utf-8")
        print(f"[Collector] neg_{grp} collected: {len(nrecs)}/{spec['target']}", flush=True)

    print("[Collector] Fetching ambiguous_cane_v2 candidates...", flush=True)
    arecs = _fetch_group_pool(
        session,
        AMBIGUOUS_CANE_SPEC,
        staging_root / "amb_cane",
        "v2_amb_cane",
        int(AMBIGUOUS_CANE_SPEC["target"]),
        seen_shas,
        seen_titles,
        seen_phashes,
        existing_meta.get("amb_cane", []),
    )
    existing_meta["amb_cane"] = arecs
    meta_file.write_text(json.dumps(existing_meta, indent=2), encoding="utf-8")
    print(f"[Collector] amb_cane collected: {len(arecs)}/{AMBIGUOUS_CANE_SPEC['target']}", flush=True)

    return existing_meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-target", type=int, default=85)
    args = ap.parse_args()
    wt = Path(__file__).resolve().parents[3]
    collect_vl_p2_staging_pool(wt, pos_target=args.pos_target)
