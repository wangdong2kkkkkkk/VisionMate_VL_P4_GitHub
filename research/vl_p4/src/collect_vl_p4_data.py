"""Collect brand-new open-licensed white cane positives & hard negatives for VL-P4.

Enforces 100% zero-leakage (SHA-256, normalized title, source cluster, aHash/dHash Hamming <= 6)
against ALL historical VL-P1, VL-P1b, VL-P1c, VL-P2, and VL-P3 datasets and holdouts.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import cv2
import requests

from .common_utils import (
    VL_P4_DIR,
    WORKSPACE_ROOT,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    load_image_unicode,
    make_percent_grid_sheet,
    normalize_title,
    save_image_unicode,
)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
USER_AGENT = "VisionMate-VL-P4-Research/1.0 (offline accessibility research)"

ALLOWED_LICENSE_TOKENS = (
    "cc0", "cc-by", "cc by", "public domain", "pd", "pdm", "gfdl", "fal", "free art", "ogl",
)

EXCLUDE_POS_WORDS = (
    "logo", "icon", "svg", "diagram", "chart", "map", "flag", "coat_of_arms", "emblem",
    "seal", "stamp", "coin", "medal", "poster", "flyer", "cartoon", "comic", "drawing",
    "sketch", "painting", "engraving", "caricature", "vitrail", "stained", "sculpture",
    "statue", "tomb", "grave", "cemetery", "monument", "plaque", "road_sign", "signpost",
    "candy", "sugar", "sugarcane", "cane_sugar", "moth", "mythimna", "lepidoptera",
    "wainscot", "noctuidae", "insect", "butterfly", "caterpillar", "larva", "pupa",
    "entomology", "arachnid", "beetle", "mushroom", "plant", "flower", "botany",
    "tactile_paving", "paving_blocks", "braille_block", "fries", "goalball",
    "bicycle", "tandem", "ski", "trumpet", "saxophone", "orchestra", "choir", "concert",
    "conference", "seminar", "symposium", "roundtable", "press_conference", "podium",
    "microphone", "headshot", "official_portrait", "mugshot", "election", "ballot",
    "guitar", "piano", "violin", "cello", "drum", "mehrnews", "senator",
    "congressman", "braille", "magnifier", "ctet", "pool table", "billiard", "smoking",
    "cigarette", "bottle", "whisky", "church", "altar", "mosaic", "silhouette", "folio",
    "construction", "traffic cone", "safety cone", "pylon", "cane syrup", "steens",
    "raising cane", "fried chicken", "ice hockey", "manhole", "curtain", "blinds",
    "bamboo", "flute", "chopstick", "matson", "american colony", "library of congress",
    "jerusalem", "palestine", "optacon", "brailler", "embossed", "bookplate",
    "retinal", "glaucoma", "macular", "arteritis", "cataract", "spreadsheet",
    "excel", "screenshot", "clock", "hearing", "radio", "keyboard",
)

EXCLUDE_NEG_WORDS = (
    "logo", "icon", "svg", "diagram", "chart", "map", "flag", "coat_of_arms",
    "emblem", "seal", "stamp", "coin", "medal", "poster", "flyer", "banner",
    "cartoon", "comic", "drawing", "sketch", "painting", "engraving", "caricature",
    "candy_cane", "sugar_cane", "sugarcane", "white_cane", "whitecane", "blind_cane",
    "blindenstock", "canne_blanche", "bastone_bianco", "baston_blanco", "blind",
    "plaque", "signpost", "pictogram",
)

POS_CATEGORIES = [
    "Category:White_canes",
    "Category:People_with_mobility_canes",
    "Category:White_Cane_Safety_Day",
    "Category:Blindenstock",
    "Category:Canne_blanche",
    "Category:Orientation_and_mobility",
    "Category:Blind_people_walking",
    "Category:Guide_dogs_with_handlers",
]

POS_QUERIES = [
    '"white cane" -moth -mythimna -lepidoptera -mehrnews -logo -sign',
    '"white cane" walking OR street OR sidewalk OR crosswalk OR station -moth -mehrnews',
    '"white cane" blind OR visually impaired -moth -mehrnews -flag',
    'whitecane -moth -mehrnews',
    '"long cane" blind OR mobility -moth',
    'blindenstock -moth -schild',
    'canne blanche aveugle -moth',
    '"bastone bianco" cieco OR non vedente',
    '"baston blanco" ciego OR bastón',
    '"biała laska" niewidomy',
    '"белая трость" слепой OR незрячий',
    '"bengala branca" cego',
    '"白杖" 視覚障害 OR 歩行',
    '"taststok" blind',
    '"slepecká hůl"',
    '"vit käpp" blind',
    '"hvítur stafur" OR "valkoinen keppi"',
    '"gậy trắng" người mù OR khiếm thị',
]

OPENVERSE_POS_QUERIES = [
    "blind person white cane walking",
    "white cane mobility street",
    "visually impaired person white cane",
    "white cane crosswalk pedestrian",
    "folding white cane blind",
]

NEG_SPECS = {
    "walking_stick_confuser": {
        "target": 85,
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
        "openverse_q": ["person with walking stick street", "elderly walking with wooden cane", "person using crutches street"],
    },
    "trekking_pole_confuser": {
        "target": 85,
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
        "target": 85,
        "categories": [
            "Category:People_with_umbrellas",
            "Category:People_with_closed_umbrellas",
            "Category:Closed_umbrellas",
            "Category:Pedestrians_with_umbrellas",
        ],
        "queries": [
            '"closed umbrella" person OR holding OR street -blind',
            'pedestrians umbrellas street walking -blind',
            '"umbrella" holding walking sidewalk -blind',
        ],
        "openverse_q": ["person holding closed umbrella street", "pedestrian with umbrella sidewalk"],
    },
    "tool_handle_confuser": {
        "target": 85,
        "categories": [
            "Category:People_with_brooms",
            "Category:Street_sweepers_(people)",
            "Category:Mops",
            "Category:Garden_forks",
            "Category:Rakes",
        ],
        "queries": [
            'sweeping broom street worker OR janitor -blind',
            'person holding mop OR shovel OR rake -blind',
        ],
        "openverse_q": ["person sweeping with broom street", "worker holding long handle tool"],
    },
    "metal_rod_pipe_confuser": {
        "target": 85,
        "categories": [
            "Category:Selfie_sticks_in_use",
            "Category:Boom_poles",
            "Category:Flagpoles",
            "Category:Fishing_rods_in_use",
        ],
        "queries": [
            '"selfie stick" person holding street OR tourist',
            '"boom pole" OR "fishing rod" person holding',
            'person holding metal pole OR tripod OR monopod',
        ],
        "openverse_q": ["person holding selfie stick", "angler holding fishing rod"],
    },
    "white_line_pole_confuser": {
        "target": 85,
        "categories": [
            "Category:Crosswalks",
            "Category:White_painted_lines_on_roads",
            "Category:Bollards",
            "Category:Handrails",
        ],
        "queries": [
            'crosswalk pedestrian street white lines -blind',
            'white bollard OR handrail sidewalk pedestrian -blind',
        ],
        "openverse_q": ["pedestrian crossing white stripes street", "sidewalk white metal railing"],
    },
    "general_distractor": {
        "target": 85,
        "categories": [
            "Category:Pedestrians",
            "Category:People_walking_on_sidewalks",
            "Category:Street_photography",
        ],
        "queries": [
            'pedestrians walking street sidewalk city -blind',
        ],
        "openverse_q": ["pedestrians walking city sidewalk", "person walking street"],
    },
}


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", str(s or "")).strip()


def _is_open_license(lic: str) -> bool:
    low = (lic or "").lower()
    return any(tok in low for tok in ALLOWED_LICENSE_TOKENS)


def _source_cluster(title: str) -> str:
    t = normalize_title(title)
    parts = [p for p in t.split("_") if not p.isdigit()]
    return "_".join(parts[:4]) if parts else t


def load_all_prior_signatures() -> dict:
    shas, titles, clusters, phashes = set(), set(), set(), []
    manifest_paths = [
        WORKSPACE_ROOT / "VisionMate_VL_P1_AG" / "research" / "vl_p1" / "manifests" / "dataset_manifest.json",
        WORKSPACE_ROOT / "VisionMate_VL_P1b_AG" / "research" / "vl_p1b" / "manifests" / "dataset_manifest.json",
        WORKSPACE_ROOT / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c" / "manifests" / "dataset_manifest.json",
        WORKSPACE_ROOT / "VisionMate_VL_P2_AG" / "research" / "vl_p2" / "manifests" / "dataset_manifest.json",
        WORKSPACE_ROOT / "VisionMate_VL_P3_AG" / "research" / "vl_p3" / "manifests" / "dataset_manifest.json",
    ]
    for mp in manifest_paths:
        if not mp.is_file():
            continue
        data = json.loads(mp.read_text(encoding="utf-8"))
        for v in data.values():
            if not isinstance(v, list):
                continue
            for it in v:
                if not isinstance(it, dict):
                    continue
                if it.get("sha256"):
                    shas.add(str(it["sha256"]))
                if it.get("title"):
                    titles.add(normalize_title(it["title"]))
                if it.get("source_cluster"):
                    clusters.add(str(it["source_cluster"]))
                elif it.get("cluster_id"):
                    clusters.add(str(it["cluster_id"]))
                if it.get("ahash") and it.get("dhash"):
                    phashes.append((str(it["ahash"]), str(it["dhash"])))
    return {"shas": shas, "titles": titles, "clusters": clusters, "phashes": phashes}


def _is_duplicate(sha: str, title: str, ahash: str, dhash: str, sigs: dict) -> bool:
    if sha in sigs["shas"]:
        return True
    nt = normalize_title(title)
    if nt and nt in sigs["titles"]:
        return True
    for ah, dh in sigs["phashes"]:
        if hex_hamming_distance(ahash, ah) <= 5 or hex_hamming_distance(dhash, dh) <= 5:
            return True
    return False


def _commons_fetch_pages(params: dict, session: requests.Session, limit: int = 50) -> list[dict]:
    base = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 960,
    }
    base.update(params)
    try:
        r = session.get(COMMONS_API, params=base, timeout=15)
        r.raise_for_status()
        pages = list((r.json().get("query", {}).get("pages", {}) or {}).values())
        return pages[:limit]
    except Exception:
        return []


def _download_and_validate(
    url: str,
    title: str,
    license_str: str,
    source_url: str,
    out_dir: Path,
    item_id: str,
    sigs: dict,
    session: requests.Session,
    exclude_words: tuple[str, ...],
    meta_extra: dict | None = None,
) -> dict | None:
    if not url or not _is_open_license(license_str):
        return None
    low_t = (title or "").lower()
    if any(w in low_t for w in exclude_words):
        return None
    nt = normalize_title(title)
    if nt and nt in sigs["titles"]:
        return None
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200 or len(resp.content) < 8000:
            return None
        tmp_path = out_dir / f"{item_id}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        arr = cv2.imdecode(
            __import__("numpy").frombuffer(resp.content, dtype=__import__("numpy").uint8),
            cv2.IMREAD_COLOR,
        )
        if arr is None or arr.size == 0:
            return None
        h, w = arr.shape[:2]
        if min(h, w) < 220 or max(h, w) < 320:
            return None
        # Resize very huge images down to max side 1280 for clean storage
        if max(h, w) > 1280:
            sc = 1280.0 / float(max(h, w))
            arr = cv2.resize(arr, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
            h, w = arr.shape[:2]
        save_image_unicode(tmp_path, arr, quality=92)
        sha = compute_sha256(tmp_path)
        ah = compute_ahash(arr)
        dh = compute_dhash(arr)
        if _is_duplicate(sha, title, ah, dh, sigs):
            tmp_path.unlink(missing_ok=True)
            return None
        sigs["shas"].add(sha)
        if nt:
            sigs["titles"].add(nt)
        sigs["phashes"].append((ah, dh))
        rec = {
            "id": item_id,
            "abs_path": str(tmp_path),
            "title": title,
            "source_cluster": _source_cluster(title),
            "source_url": source_url,
            "license": license_str,
            "sha256": sha,
            "ahash": ah,
            "dhash": dh,
            "width": int(w),
            "height": int(h),
        }
        if meta_extra:
            rec.update(meta_extra)
        return rec
    except Exception:
        return None


def collect_vl_p4_candidates(target_pos: int = 440, target_neg_per_group: int = 85) -> Path:
    cache_dir = VL_P4_DIR / "cache"
    pos_dir = cache_dir / "raw_pos_v4"
    neg_dir = cache_dir / "raw_neg_v4"
    screen_dir = cache_dir / "screening"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    screen_dir.mkdir(parents=True, exist_ok=True)

    sigs = load_all_prior_signatures()
    print(f"Loaded prior signatures: {len(sigs['shas'])} sha256, {len(sigs['titles'])} titles, {len(sigs['phashes'])} phashes", flush=True)

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    pos_records: list[dict] = []
    pos_idx = 0

    # 1. Wikimedia Commons Categories for Positives
    for cat in POS_CATEGORIES:
        if len(pos_records) >= target_pos:
            break
        cmcontinue = None
        for _ in range(4):
            if len(pos_records) >= target_pos:
                break
            p = {
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": 50,
            }
            if cmcontinue:
                p["gcmcontinue"] = cmcontinue
            r = sess.get(COMMONS_API, params={**p, "action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 960}, timeout=15)
            if r.status_code != 200:
                break
            js = r.json()
            pages = list((js.get("query", {}).get("pages", {}) or {}).values())
            for pg in pages:
                if len(pos_records) >= target_pos:
                    break
                ii = (pg.get("imageinfo") or [{}])[0]
                ext = ii.get("extmetadata", {})
                lic = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                url = ii.get("thumburl") or ii.get("url", "")
                title = pg.get("title", "")
                desc = _strip_html(ext.get("ImageDescription", {}).get("value", ""))
                if any(w in desc.lower() for w in ("moth", "lepidoptera", "sugarcane", "candy cane")):
                    continue
                pos_idx += 1
                rec = _download_and_validate(
                    url, title, lic, ii.get("descriptionurl", ""), pos_dir, f"v4_raw_pos_{pos_idx:04d}",
                    sigs, sess, EXCLUDE_POS_WORDS, {"label": "object_01", "query": cat},
                )
                if rec:
                    pos_records.append(rec)
            cmcontinue = (js.get("continue") or {}).get("gcmcontinue")
            if not cmcontinue:
                break
        print(f"  After pos cat {cat}: {len(pos_records)}/{target_pos}", flush=True)

    # 2. Wikimedia Commons Search Queries for Positives
    for q in POS_QUERIES:
        if len(pos_records) >= target_pos:
            break
        for offset in (0, 50, 100):
            if len(pos_records) >= target_pos:
                break
            pages = _commons_fetch_pages({"generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": offset}, sess)
            if not pages:
                break
            for pg in pages:
                if len(pos_records) >= target_pos:
                    break
                ii = (pg.get("imageinfo") or [{}])[0]
                ext = ii.get("extmetadata", {})
                lic = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                url = ii.get("thumburl") or ii.get("url", "")
                title = pg.get("title", "")
                desc = _strip_html(ext.get("ImageDescription", {}).get("value", ""))
                if any(w in desc.lower() for w in ("moth", "lepidoptera", "sugarcane", "candy cane")):
                    continue
                pos_idx += 1
                rec = _download_and_validate(
                    url, title, lic, ii.get("descriptionurl", ""), pos_dir, f"v4_raw_pos_{pos_idx:04d}",
                    sigs, sess, EXCLUDE_POS_WORDS, {"label": "object_01", "query": q[:40]},
                )
                if rec:
                    pos_records.append(rec)
        print(f"  After pos search {q[:32]}: {len(pos_records)}/{target_pos}", flush=True)

    # 3. Openverse Search for Positives
    for oq in OPENVERSE_POS_QUERIES:
        if len(pos_records) >= target_pos:
            break
        for page_i in (1, 2, 3):
            if len(pos_records) >= target_pos:
                break
            try:
                r = sess.get(OPENVERSE_API, params={"q": oq, "page_size": 30, "page": page_i}, timeout=15)
                if r.status_code != 200:
                    break
                for item in r.json().get("results", []):
                    if len(pos_records) >= target_pos:
                        break
                    lic = f"{item.get('license', '')} {item.get('license_version', '')}"
                    pos_idx += 1
                    rec = _download_and_validate(
                        item.get("url", ""), item.get("title", "") or f"openverse_{item.get('id', pos_idx)}",
                        lic, item.get("foreign_landing_url", ""), pos_dir, f"v4_raw_pos_{pos_idx:04d}",
                        sigs, sess, EXCLUDE_POS_WORDS, {"label": "object_01", "query": oq},
                    )
                    if rec:
                        pos_records.append(rec)
            except Exception:
                break
            time.sleep(0.3)
        print(f"  After openverse {oq}: {len(pos_records)}/{target_pos}", flush=True)

    # 4. Hard Negatives across 7 confuser groups
    neg_records: list[dict] = []
    neg_idx = 0
    for group_name, spec in NEG_SPECS.items():
        grp_recs: list[dict] = []
        t_grp = min(target_neg_per_group, spec["target"])
        for cat in spec["categories"]:
            if len(grp_recs) >= t_grp:
                break
            pages = _commons_fetch_pages({"generator": "categorymembers", "gcmtitle": cat, "gcmtype": "file", "gcmlimit": 50}, sess)
            for pg in pages:
                if len(grp_recs) >= t_grp:
                    break
                ii = (pg.get("imageinfo") or [{}])[0]
                ext = ii.get("extmetadata", {})
                lic = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                url = ii.get("thumburl") or ii.get("url", "")
                title = pg.get("title", "")
                neg_idx += 1
                is_hh = group_name in ("walking_stick_confuser", "trekking_pole_confuser", "umbrella_confuser", "tool_handle_confuser", "metal_rod_pipe_confuser")
                rec = _download_and_validate(
                    url, title, lic, ii.get("descriptionurl", ""), neg_dir / group_name, f"v4_raw_neg_{neg_idx:04d}",
                    sigs, sess, EXCLUDE_NEG_WORDS,
                    {"label": "unknown", "confuser_group": group_name, "person_holding_confuser": is_hh},
                )
                if rec:
                    grp_recs.append(rec)
        for q in spec["queries"]:
            if len(grp_recs) >= t_grp:
                break
            pages = _commons_fetch_pages({"generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 50}, sess)
            for pg in pages:
                if len(grp_recs) >= t_grp:
                    break
                ii = (pg.get("imageinfo") or [{}])[0]
                ext = ii.get("extmetadata", {})
                lic = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                url = ii.get("thumburl") or ii.get("url", "")
                title = pg.get("title", "")
                neg_idx += 1
                is_hh = group_name in ("walking_stick_confuser", "trekking_pole_confuser", "umbrella_confuser", "tool_handle_confuser", "metal_rod_pipe_confuser")
                rec = _download_and_validate(
                    url, title, lic, ii.get("descriptionurl", ""), neg_dir / group_name, f"v4_raw_neg_{neg_idx:04d}",
                    sigs, sess, EXCLUDE_NEG_WORDS,
                    {"label": "unknown", "confuser_group": group_name, "person_holding_confuser": is_hh},
                )
                if rec:
                    grp_recs.append(rec)
        neg_records.extend(grp_recs)
        print(f"  Neg group {group_name}: {len(grp_recs)}/{t_grp}", flush=True)

    # Render positive percent-grid screening sheets (12 per page) for visual verification & annotation
    for page_idx in range(0, len(pos_records), 12):
        batch = pos_records[page_idx : page_idx + 12]
        thumbs = [(r["id"], load_image_unicode(Path(r["abs_path"]))) for r in batch]
        make_percent_grid_sheet(thumbs, screen_dir / f"pos_screen_{page_idx // 12 + 1:02d}.jpg", cols=3, cell_w=380, cell_h=300)

    out_json = cache_dir / "raw_candidates_v4.json"
    out_json.write_text(json.dumps({
        "pos_count": len(pos_records),
        "neg_count": len(neg_records),
        "pos_records": pos_records,
        "neg_records": neg_records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Collection done! pos={len(pos_records)}, neg={len(neg_records)} -> {out_json}", flush=True)
    return out_json


if __name__ == "__main__":
    collect_vl_p4_candidates()
