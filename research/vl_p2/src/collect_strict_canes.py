"""Dedicated strict Wikimedia Commons white-cane collector.

Avoids broad categories (`Guide_dogs`, `Blindness`, `Tactile_paving`) that return conference photos
and tactile tiles without canes. Queries genuine white-cane categories + multilingual queries and
enforces metadata cane-keyword checks before downloading.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import requests

from .collect_vl_p2_dataset import (
    COMMONS_API,
    USER_AGENT,
    _api_fetch_paginated,
    _download_candidate,
    _is_duplicate,
    _load_all_prior_signatures,
)
from .common_utils import normalize_title

CANE_KEYWORDS = (
    "white cane", "white_cane", "white-cane", "long cane", "long_cane", "mobility cane",
    "blind cane", "folding cane", "hoover cane", "canne blanche", "canne_blanche",
    "blindenstock", "langstock", "blinden-stock", "taststock", "bastón blanco",
    "baston blanco", "baston_blanco", "bastón_blanco", "bengala branca", "bengala_branca",
    "bastone bianco", "bastone_bianco", "biała laska", "biala laska", "białą laską",
    "белая трость", "белой тростью", "白杖", "blindenstok", "blindkäpp", "vit käpp",
    "hvid stok", "blind", "visually impaired", "visually_impaired",
)

BAD_META_WORDS = (
    "speech", "conference", "symposium", "microphone", "concert", "choir", "orchestra",
    "congressman", "senator", "ambassador", "candy", "sugar", "tactile tile",
    "tactile paving only", "closeup", "braille display", "brailler", "perkins",
    "lego", "u+1f9af", "emoji", "robot dog", "qr code", "signpost",
)

STRICT_CANE_CATEGORIES = [
    "Category:White_canes",
    "Category:People_with_white_canes",
    "Category:White_Cane_Safety_Day",
]

STRICT_CANE_QUERIES = [
    '"white cane" -candy -sugar -logo -icon -svg',
    'intitle:white_cane OR intitle:"white cane" -candy -sugar',
    '"white cane" walking OR crossing OR street OR sidewalk',
    '"white cane" red tip OR folding OR mobility',
    '"canne blanche" -signe -icon -logo',
    '"Blindenstock" OR "Langstock" OR "Taststock"',
    '"bastón blanco" OR "baston blanco"',
    '"bengala branca"',
    '"bastone bianco" cieco OR non vedente OR strisce',
    '"biała laska" OR "białą laską"',
    '"белая трость" OR "белой тростью"',
    '"白杖"',
    '"blindenstok" OR "blindegeleidestok"',
    '"blindkäpp" OR "vit käpp"',
    '"long cane" orientation mobility',
    '"blind person" cane walking street',
    '"visually impaired" cane walking',
]


def _meta_looks_like_real_cane(cand: dict) -> bool:
    blob = " ".join([
        cand.get("title", ""),
        cand.get("description", ""),
        cand.get("categories", ""),
    ]).lower()
    if any(b in blob for b in BAD_META_WORDS):
        return False
    # In guide dog categories without cane mention, reject
    if "guide dog" in blob and not any(k in blob for k in ("cane", "canne", "stock", "baston", "bastón", "bengala", "bastone", "lask", "трост", "杖", "stok")):
        return False
    return any(k in blob for k in CANE_KEYWORDS)


def collect_strict_cane_pool(worktree_root: Path, target_count: int = 110) -> list[dict]:
    """Download strictly screened open-licensed white-cane photos into cache/staging_new/pos_strict_canes."""
    ai_root = worktree_root.parent
    vl_p1_dir = ai_root / "VisionMate_VL_P1_AG" / "research" / "vl_p1"
    vl_p1b_dir = ai_root / "VisionMate_VL_P1b_AG" / "research" / "vl_p1b"
    vl_p1c_dir = ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"
    vl_p2_dir = worktree_root / "research" / "vl_p2"

    seen_shas, seen_titles, _, seen_phashes = _load_all_prior_signatures(
        vl_p1_dir, vl_p1b_dir, vl_p1c_dir
    )

    staging_root = vl_p2_dir / "cache" / "staging_new"
    out_dir = staging_root / "pos_strict_canes"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file = staging_root / "strict_canes_metadata.json"

    recs: list[dict] = []
    if meta_file.is_file():
        recs = [r for r in json.loads(meta_file.read_text(encoding="utf-8")) if Path(r["abs_path"]).is_file()]
        for r in recs:
            seen_shas.add(r["sha256"])
            seen_titles.add(r["norm_title"])
            seen_phashes.append((r["id"], r["ahash"], r["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    candidates: list[dict] = []
    batch_titles: set[str] = set()

    for cat in STRICT_CANE_CATEGORIES:
        rows = _api_fetch_paginated(session, {
            "generator": "categorymembers",
            "gcmtitle": cat,
            "gcmtype": "file",
            "gcmlimit": 50,
        }, max_pages=6)
        for row in rows:
            nt = row["norm_title"]
            if nt not in seen_titles and nt not in batch_titles and _meta_looks_like_real_cane(row):
                batch_titles.add(nt)
                candidates.append(row)
        time.sleep(0.25)

    for q in STRICT_CANE_QUERIES:
        if len(candidates) >= target_count * 4:
            break
        rows = _api_fetch_paginated(session, {
            "generator": "search",
            "gsrsearch": q,
            "gsrnamespace": 6,
            "gsrlimit": 50,
        }, max_pages=4)
        for row in rows:
            nt = row["norm_title"]
            if nt not in seen_titles and nt not in batch_titles and _meta_looks_like_real_cane(row):
                batch_titles.add(nt)
                candidates.append(row)
        time.sleep(0.25)

    print(f"[StrictCaneCollector] Metadata-screened candidate pool size: {len(candidates)}", flush=True)
    next_idx = len(recs) + 1
    for cand in candidates:
        if len(recs) >= target_count:
            break
        out_p = out_dir / f"v2_scane_{next_idx:03d}.jpg"
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
            "id": f"v2_scane_{next_idx:03d}",
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
        meta_file.write_text(json.dumps(recs, indent=2), encoding="utf-8")
        next_idx += 1
        time.sleep(0.12)

    print(f"[StrictCaneCollector] Final downloaded count: {len(recs)}/{target_count}", flush=True)
    return recs


if __name__ == "__main__":
    wt = Path(__file__).resolve().parents[3]
    collect_strict_cane_pool(wt, target_count=110)
