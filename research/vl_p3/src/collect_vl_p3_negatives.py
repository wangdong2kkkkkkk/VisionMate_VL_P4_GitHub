"""Collect brand-new open-licensed hard negatives (especially handheld confusers) & screen unused positives for VL-P3.

Enforces strict zero-leakage (SHA-256, normalized title, source cluster, aHash/dHash Hamming > 6)
against ALL historical VL-P1, VL-P1b, VL-P1c, and VL-P2 manifests/data caches.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import requests

from .common_utils import (
    VL_P3_DIR,
    WORKTREE_ROOT,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    load_image_unicode,
    normalize_title,
    save_image_unicode,
)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
USER_AGENT = "VisionMate-VL-P3-Research/1.0 (offline accessibility research; https://example.org)"

ALLOWED_LICENSE_TOKENS = (
    "cc0",
    "cc-by",
    "cc by",
    "public domain",
    "pd",
    "pdm",
    "gfdl",
    "fal",
    "free art",
    "ogl",
)

EXCLUDE_TITLE_WORDS = (
    "logo", "icon", "svg", "diagram", "chart", "map", "flag", "coat_of_arms",
    "emblem", "seal", "stamp", "coin", "medal", "poster", "flyer", "banner",
    "cartoon", "comic", "drawing", "sketch", "painting", "engraving", "caricature",
    "candy_cane", "sugar_cane", "sugarcane", "white_cane", "whitecane", "blind_cane",
    "blindenstock", "canne_blanche", "bastone_bianco", "baston_blanco", "blind",
    "plaque", "signpost", "pictogram",
)

HANDHELD_POSITIVE_HINTS = (
    "man ", "woman", "person", "pedestrian", "hiker", "walking", "holding", "elderly",
    "senior", "tourist", "traveller", "traveler", "climber", "nordic", "rain", "street",
    "worker", "janitor", "cleaner", "farmer", "athlete", "ski",
)

VL_P3_NEG_SPECS = {
    "walking_stick_confuser": {
        "target": 48,
        "categories": [
            "Category:People_with_walking_sticks",
            "Category:Walking_sticks",
            "Category:Hikers_with_walking_sticks",
            "Category:Crutches_in_use",
        ],
        "queries": [
            '"walking stick" person OR hiker OR man OR woman -blind -white_cane',
            '"walking cane" man OR woman OR elderly sidewalk OR street -blind -white',
            '"crutches" walking street OR person -blind',
        ],
        "openverse_q": ["person with walking stick street", "elderly walking with wooden cane"],
    },
    "trekking_pole_confuser": {
        "target": 48,
        "categories": [
            "Category:Hikers_with_trekking_poles",
            "Category:Trekking_poles",
            "Category:Nordic_walking",
            "Category:Skiers_holding_ski_poles",
        ],
        "queries": [
            '"trekking poles" hiker OR person OR walking trail -blind',
            '"Nordic walking" person OR sticks OR poles street OR park',
            '"hiking poles" trail walking -blind',
        ],
        "openverse_q": ["hiker with trekking poles", "nordic walking sticks"],
    },
    "umbrella_confuser": {
        "target": 52,
        "categories": [
            "Category:People_with_umbrellas",
            "Category:People_with_closed_umbrellas",
            "Category:Closed_umbrellas",
            "Category:Pedestrians_with_umbrellas",
        ],
        "queries": [
            '"umbrella" walking street OR pedestrian OR man OR woman rain -white_cane -blind',
            '"closed umbrella" OR "umbrella" holding hand street -blind',
            '"parasol" OR "umbrella" pedestrian sidewalk -cane -blind',
        ],
        "openverse_q": ["person holding closed umbrella walking", "pedestrian with umbrella street"],
    },
    "metal_rod_pipe_confuser": {
        "target": 42,
        "categories": [
            "Category:Monopods",
            "Category:People_using_selfie_sticks",
            "Category:Fishing_rods_in_use",
            "Category:Selfie_sticks",
        ],
        "queries": [
            '"selfie stick" person OR tourist holding -blind',
            '"monopod" photographer OR tripod holding -blind',
            '"metal pipe" OR "steel rod" OR "aluminum pole" holding OR worker -cane -blind',
        ],
        "openverse_q": ["person holding selfie stick", "photographer with monopod"],
    },
    "tool_handle_confuser": {
        "target": 40,
        "categories": [
            "Category:Brooms_in_use",
            "Category:Mops",
            "Category:Street_sweepers",
            "Category:Rakes",
        ],
        "queries": [
            '"broom" street sweeper OR sweeping OR handle -cane -blind',
            '"mop" OR "long handle" worker OR floor cleaning -cane -blind',
            '"rake" OR "shovel" holding garden handle -cane -blind',
        ],
        "openverse_q": ["street sweeper broom handle", "person holding mop or broom"],
    },
    "white_line_pole_confuser": {
        "target": 40,
        "categories": [
            "Category:Bollards",
            "Category:Zebra_crossings",
            "Category:Safety_handrails",
            "Category:Road_markings",
        ],
        "queries": [
            '"white bollard" OR "white post" sidewalk OR street -blind -cane',
            '"zebra crossing" street white stripes pedestrian -cane -blind',
            '"handrail" white staircase OR ramp outdoor -cane -blind',
        ],
        "openverse_q": ["white bollard street sidewalk", "white railing ramp pedestrian"],
    },
    "general_distractor": {
        "target": 45,
        "categories": [
            "Category:Pedestrians_crossing_streets",
            "Category:Sidewalks",
            "Category:Subway_platforms",
            "Category:Crosswalks",
        ],
        "queries": [
            '"pedestrians" sidewalk city walking -cane -blind -umbrella',
            '"crosswalk" people walking urban street -cane -blind',
            '"platform" commuters station standing -cane -blind',
        ],
        "openverse_q": ["pedestrians walking on sidewalk", "people crossing city street"],
    },
}

POSITIVE_EXTRA_QUERIES = [
    '"white cane" crossing OR sidewalk OR pedestrian',
    '"white cane" red tip OR mobility cane',
    '"canne blanche" rue OR trottoir OR pieton',
    '"Blindenstock" Straße OR Fussgänger OR Überweg',
    '"bastón blanco" calle OR peatón',
    '"白杖" 歩行 OR 横断歩道',
]


def _is_open_license(lic: str) -> bool:
    low = (lic or "").strip().lower()
    if not low or "non-commercial" in low or "-nc" in low or "nc-" in low or "fair use" in low:
        return False
    return any(tok in low for tok in ALLOWED_LICENSE_TOKENS)


def _strip_html(raw: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", raw or "").split())


def _is_clean_title(title: str, allow_cane: bool = False) -> bool:
    tl = title.lower()
    if not tl.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    bad = EXCLUDE_TITLE_WORDS if not allow_cane else tuple(w for w in EXCLUDE_TITLE_WORDS if "cane" not in w and "blind" not in w and "stock" not in w and "blanche" not in w and "baston" not in w)
    return not any(b in tl for b in bad)


def load_all_prior_signatures() -> tuple[set[str], set[str], set[str], list[tuple[str, str, str]]]:
    """Gather all SHA-256, normalized titles, clusters, and pHashes across VL-P1/P1b/P1c/P2."""
    ai_root = WORKTREE_ROOT.parent
    rdirs = [
        ai_root / "VisionMate_VL_P1_AG" / "research" / "vl_p1",
        ai_root / "VisionMate_VL_P1b_AG" / "research" / "vl_p1b",
        ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c",
        ai_root / "VisionMate_VL_P2_AG" / "research" / "vl_p2",
        VL_P3_DIR,
    ]
    seen_shas: set[str] = set()
    seen_titles: set[str] = set()
    seen_clusters: set[str] = set()
    seen_phashes: list[tuple[str, str, str]] = []

    for rdir in rdirs:
        if not rdir.exists():
            continue
        for mf_name in ("dataset_manifest.json",):
            mf_p = rdir / "manifests" / mf_name
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
                                seen_shas.add(str(rec[sk]).lower())
                        for tk in ("source_title", "title", "norm_title"):
                            if rec.get(tk):
                                seen_titles.add(normalize_title(rec[tk]))
                        if rec.get("source_cluster"):
                            seen_clusters.add(normalize_title(rec["source_cluster"]))
                        if rec.get("ahash") and rec.get("dhash"):
                            seen_phashes.append((str(rec.get("id", key)), rec["ahash"], rec["dhash"]))
        data_root = rdir / "data"
        if data_root.exists():
            for p in data_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    seen_shas.add(compute_sha256(p).lower())
        cache_data = rdir / "cache" / "data"
        if cache_data.exists():
            for p in cache_data.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    seen_shas.add(compute_sha256(p).lower())
    return seen_shas, seen_titles, seen_clusters, seen_phashes


def _is_duplicate(sha: str, norm_t: str, cluster: str, ah: str, dh: str, seen_shas: set[str], seen_titles: set[str], seen_clusters: set[str], seen_phashes: list[tuple[str, str, str]]) -> bool:
    if sha.lower() in seen_shas or (norm_t and norm_t in seen_titles) or (cluster and cluster in seen_clusters):
        return True
    for _, o_ah, o_dh in seen_phashes:
        if hex_hamming_distance(ah, o_ah) <= 6 or hex_hamming_distance(dh, o_dh) <= 6:
            return True
    return False


def _commons_fetch(session: requests.Session, params: dict, allow_cane: bool = False, max_pages: int = 2) -> list[dict]:
    out = []
    cur = dict(params)
    for _ in range(max_pages):
        req = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 960,
            "format": "json",
        }
        req.update(cur)
        data = None
        for attempt in range(3):
            try:
                r = session.get(COMMONS_API, params=req, timeout=20)
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
            if not _is_clean_title(title, allow_cane=allow_cane):
                continue
            iis = page.get("imageinfo", [])
            if not iis:
                continue
            ii = iis[0]
            if ii.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            w = int(ii.get("thumbwidth") or ii.get("width") or 0)
            h = int(ii.get("thumbheight") or ii.get("height") or 0)
            if min(w, h) < 220:
                continue
            meta = ii.get("extmetadata", {})
            lic = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not _is_open_license(lic):
                continue
            desc = _strip_html(meta.get("ImageDescription", {}).get("value", ""))[:240]
            cats = _strip_html(meta.get("Categories", {}).get("value", ""))[:200]
            nt = normalize_title(title)
            out.append({
                "title": title,
                "norm_title": nt,
                "source_cluster": " ".join(nt.split()[:4]) if nt else nt,
                "page_url": ii.get("descriptionurl", ""),
                "image_url": ii.get("thumburl") or ii.get("url"),
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


def _openverse_fetch(session: requests.Session, query: str, page_size: int = 30) -> list[dict]:
    out = []
    try:
        r = session.get(
            OPENVERSE_API,
            params={"q": query, "page_size": page_size, "license_type": "commercial"},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        for row in data.get("results", []):
            lic = f"{row.get('license', '')} {row.get('license_version', '')}".strip()
            if not _is_open_license(lic):
                continue
            title = str(row.get("title") or row.get("id") or "")
            if any(w in title.lower() for w in ("cane", "blind", "candy", "sugar", "logo", "icon", "svg")):
                continue
            nt = normalize_title(title)
            out.append({
                "title": f"Openverse:{title[:100]}",
                "norm_title": nt,
                "source_cluster": " ".join(nt.split()[:4]) if nt else row.get("id", ""),
                "page_url": row.get("foreign_landing_url") or row.get("url") or "",
                "image_url": row.get("url"),
                "license": lic.upper(),
                "author": str(row.get("creator") or "Unknown")[:100],
                "description": title[:200],
                "categories": str(row.get("source") or "openverse"),
            })
    except Exception:
        pass
    return out


def _download_image(session: requests.Session, cand: dict, out_path: Path) -> dict | None:
    try:
        r = session.get(cand["image_url"], timeout=20)
        r.raise_for_status()
        blob = r.content
        if len(blob) < 4000:
            return None
        tmp_path = out_path.with_suffix(".jpg")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(blob)
        img = load_image_unicode(tmp_path)
        h, w = img.shape[:2]
        if min(h, w) < 220:
            tmp_path.unlink(missing_ok=True)
            return None
        if max(h, w) > 960:
            sc = 960.0 / float(max(h, w))
            img = cv2.resize(img, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
        save_image_unicode(tmp_path, img)
        sha = compute_sha256(tmp_path).lower()
        ah = compute_ahash(img)
        dh = compute_dhash(img)
        blob_txt = f"{cand.get('title', '')} {cand.get('description', '')} {cand.get('categories', '')}".lower()
        handheld_hint = any(tok in blob_txt for tok in HANDHELD_POSITIVE_HINTS)
        return {
            "abs_path": str(tmp_path),
            "sha256": sha,
            "ahash": ah,
            "dhash": dh,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "person_holding_confuser": bool(handheld_hint),
            **cand,
        }
    except Exception:
        return None


def collect_vl_p3_negatives() -> Path:
    """Download fresh hard negatives for VL-P3 across 7 confuser categories into cache/staging_v3."""
    staging_v3 = VL_P3_DIR / "cache" / "staging_v3"
    staging_v3.mkdir(parents=True, exist_ok=True)
    meta_path = staging_v3 / "v3_negatives_metadata.json"
    existing = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    seen_shas, seen_titles, seen_clusters, seen_phashes = load_all_prior_signatures()
    # Also register existing staging_v3 files
    for grp, rows in existing.items():
        for rec in rows:
            if Path(rec.get("abs_path", "")).is_file():
                seen_shas.add(rec["sha256"].lower())
                if rec.get("norm_title"):
                    seen_titles.add(rec["norm_title"])
                if rec.get("source_cluster"):
                    seen_clusters.add(normalize_title(rec["source_cluster"]))
                if rec.get("ahash") and rec.get("dhash"):
                    seen_phashes.append((rec["id"], rec["ahash"], rec["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: dict[str, list[dict]] = {}
    for grp, spec in VL_P3_NEG_SPECS.items():
        out_dir = staging_v3 / f"neg_{grp}"
        recs = [r for r in existing.get(grp, []) if Path(r.get("abs_path", "")).is_file()]
        target = spec["target"]
        if len(recs) < target:
            cands = []
            seen_cand_titles = set()
            for cat in spec["categories"]:
                if len(cands) >= target * 3:
                    break
                for row in _commons_fetch(session, {"generator": "categorymembers", "gcmtitle": cat, "gcmtype": "file", "gcmlimit": 50}, allow_cane=False, max_pages=2):
                    if row["norm_title"] not in seen_titles and row["norm_title"] not in seen_cand_titles:
                        seen_cand_titles.add(row["norm_title"])
                        cands.append(row)
            for q in spec["queries"]:
                if len(cands) >= target * 4:
                    break
                for row in _commons_fetch(session, {"generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 50}, allow_cane=False, max_pages=2):
                    if row["norm_title"] not in seen_titles and row["norm_title"] not in seen_cand_titles:
                        seen_cand_titles.add(row["norm_title"])
                        cands.append(row)
            for oq in spec.get("openverse_q", []):
                if len(cands) >= target * 5:
                    break
                for row in _openverse_fetch(session, oq, page_size=25):
                    if row["norm_title"] not in seen_titles and row["norm_title"] not in seen_cand_titles:
                        seen_cand_titles.add(row["norm_title"])
                        cands.append(row)

            # Prioritize candidates that mention people/pedestrians holding objects first
            cands.sort(key=lambda c: int(any(t in f"{c['title']} {c['description']}".lower() for t in HANDHELD_POSITIVE_HINTS)), reverse=True)
            next_idx = len(recs) + 1
            for cand in cands:
                if len(recs) >= target:
                    break
                rid = f"v3_neg_{grp[:8]}_{next_idx:03d}"
                dl = _download_image(session, cand, out_dir / f"{rid}.jpg")
                if dl is None:
                    continue
                if _is_duplicate(dl["sha256"], dl["norm_title"], normalize_title(dl["source_cluster"]), dl["ahash"], dl["dhash"], seen_shas, seen_titles, seen_clusters, seen_phashes):
                    Path(dl["abs_path"]).unlink(missing_ok=True)
                    continue
                dl["id"] = rid
                dl["confuser_group"] = grp
                seen_shas.add(dl["sha256"])
                seen_titles.add(dl["norm_title"])
                seen_clusters.add(normalize_title(dl["source_cluster"]))
                seen_phashes.append((rid, dl["ahash"], dl["dhash"]))
                recs.append(dl)
                next_idx += 1
        results[grp] = recs
        print(f"Group {grp}: {len(recs)} / {target} images ready (handheld hints: {sum(1 for r in recs if r.get('person_holding_confuser'))})")
        meta_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return meta_path


if __name__ == "__main__":
    p = collect_vl_p3_negatives()
    print("Saved VL-P3 negatives staging metadata:", p)
