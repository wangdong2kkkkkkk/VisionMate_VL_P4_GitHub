"""Diverse Multilingual White-Cane & Ambiguous White-Stick Supplementary Collector.

Caps images per author/event (`max_per_author=2`, `max_per_prefix=2`) so `final_holdout_v2`
contains truly diverse independent scenes across languages and countries rather than
one event album.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import requests

from .collect_vl_p2_dataset import (
    USER_AGENT,
    _api_fetch_paginated,
    _download_candidate,
    _is_duplicate,
    _load_all_prior_signatures,
)
from .common_utils import load_image_unicode, make_contact_sheet, normalize_title

BAD_SEARCH_WORDS = (
    "brisbane", "mehrnews", "conference", "symposium", "speech", "microphone",
    "concert", "choir", "senator", "candy", "sugar", "logo", "icon", "diagram",
    "emoji", "u+1f9af", "comic", "cartoon", "lego", "robot dog", "qr code",
)

MULTILINGUAL_QUERIES = [
    '"white cane" walking OR crossing OR pedestrian OR street -Brisbane -candy',
    '"white cane" blind street OR train OR bus OR stairs -Brisbane',
    '"white cane" folded OR folding OR tip OR mobility',
    '"Blindenstock" OR "Langstock" OR "Taststock"',
    '"canne blanche" aveugle OR rue OR pieton',
    '"bastón blanco" OR "baston blanco" ciego OR calle OR persona',
    '"bengala branca" cego OR rua OR pessoa',
    '"bastone bianco" cieco OR strada OR attraversamento',
    '"biała laska" OR "białą laską"',
    '"белая трость" OR "белой тростью"',
    '"白杖" 歩行 OR 点字 OR 電車',
    '"blindenstok" OR "blindegeleidestok"',
    '"blindkäpp" OR "vit käpp"',
    '"hvid stok" blind',
    '"white cane" subway OR metro OR platform OR curb',
]

AMBIGUOUS_WHITE_STICK_QUERIES = [
    '"PVC pipe" white OR fitting OR tube',
    '"twirling baton" OR "majorette baton" white',
    '"tent pole" fiberglass OR white',
    '"white stick" OR "white rod" OR "white dowel" OR "antenna whip" white',
]


def _prefix16(norm_t: str) -> str:
    parts = norm_t.split()
    return " ".join(parts[:3]) if len(parts) >= 3 else norm_t[:18]


def _looks_diverse_cane(cand: dict) -> bool:
    blob = " ".join([
        cand.get("title", ""),
        cand.get("description", ""),
        cand.get("categories", ""),
    ]).lower()
    if any(b in blob for b in BAD_SEARCH_WORDS):
        return False
    return True


def collect_diverse_supplement(worktree_root: Path, pos_target: int = 80, amb_target: int = 18) -> dict:
    ai_root = worktree_root.parent
    vl_p1_dir = ai_root / "VisionMate_VL_P1_AG" / "research" / "vl_p1"
    vl_p1b_dir = ai_root / "VisionMate_VL_P1b_AG" / "research" / "vl_p1b"
    vl_p1c_dir = ai_root / "VisionMate_VL_P1c_AG" / "research" / "vl_p1c"
    vl_p2_dir = worktree_root / "research" / "vl_p2"

    seen_shas, seen_titles, _, seen_phashes = _load_all_prior_signatures(
        vl_p1_dir, vl_p1b_dir, vl_p1c_dir
    )
    staging_root = vl_p2_dir / "cache" / "staging_new"
    # Also exclude already staged files so zero duplicates happen across staging folders
    for mname in ("staging_metadata.json", "strict_canes_metadata.json"):
        mp = staging_root / mname
        if mp.is_file():
            data = json.loads(mp.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else [r for lst in data.values() for r in lst]
            for r in rows:
                seen_shas.add(r["sha256"])
                seen_titles.add(r["norm_title"])
                seen_phashes.append((r["id"], r["ahash"], r["dhash"]))

    div_dir = staging_root / "pos_diverse_canes"
    div_dir.mkdir(parents=True, exist_ok=True)
    meta_out = staging_root / "diverse_canes_metadata.json"

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    author_counts: dict[str, int] = {}
    prefix_counts: dict[str, int] = {}
    pos_recs: list[dict] = []
    if meta_out.is_file():
        old = json.loads(meta_out.read_text(encoding="utf-8"))
        pos_recs = [r for r in old.get("pos_diverse", []) if Path(r["abs_path"]).is_file()]
        for r in pos_recs:
            seen_shas.add(r["sha256"])
            seen_titles.add(r["norm_title"])
            seen_phashes.append((r["id"], r["ahash"], r["dhash"]))

    candidates: list[dict] = []
    batch_titles: set[str] = set()
    for q in MULTILINGUAL_QUERIES:
        rows = _api_fetch_paginated(session, {
            "generator": "search",
            "gsrsearch": q,
            "gsrnamespace": 6,
            "gsrlimit": 50,
        }, max_pages=2)
        for row in rows:
            nt = row["norm_title"]
            if nt not in seen_titles and nt not in batch_titles and _looks_diverse_cane(row):
                batch_titles.add(nt)
                candidates.append(row)
        time.sleep(0.18)

    print(f"[DiverseCollector] Diverse cane metadata pool: {len(candidates)}", flush=True)
    next_i = len(pos_recs) + 1
    for cand in candidates:
        if len(pos_recs) >= pos_target:
            break
        auth = normalize_title(cand.get("author", "unknown"))[:32]
        pref = _prefix16(cand["norm_title"])
        if author_counts.get(auth, 0) >= 2 and auth != "unknown":
            continue
        if prefix_counts.get(pref, 0) >= 2:
            continue
        out_p = div_dir / f"v2_dcane_{next_i:03d}.jpg"
        dl = _download_candidate(session, cand, out_p)
        if dl is None:
            continue
        if _is_duplicate(dl["sha256"], dl["norm_title"], dl["ahash"], dl["dhash"], seen_shas, seen_titles, seen_phashes):
            dl["path"].unlink(missing_ok=True)
            continue
        seen_shas.add(dl["sha256"])
        seen_titles.add(dl["norm_title"])
        seen_phashes.append((dl["path"].name, dl["ahash"], dl["dhash"]))
        author_counts[auth] = author_counts.get(auth, 0) + 1
        prefix_counts[pref] = prefix_counts.get(pref, 0) + 1
        pos_recs.append({
            "id": f"v2_dcane_{next_i:03d}",
            "abs_path": str(dl["path"]),
            "sha256": dl["sha256"],
            "ahash": dl["ahash"],
            "dhash": dl["dhash"],
            "width": dl["width"],
            "height": dl["height"],
            "title": dl["title"],
            "norm_title": dl["norm_title"],
            "source_cluster": pref,
            "source_url": dl["page_url"],
            "image_url": dl["image_url"],
            "license": dl["license"],
            "author": dl["author"],
            "description": dl["description"],
        })
        next_i += 1
        time.sleep(0.08)

    # Fetch clean ambiguous white sticks / batons / PVC pipes
    amb_dir = staging_root / "amb_white_sticks"
    amb_dir.mkdir(parents=True, exist_ok=True)
    amb_cands: list[dict] = []
    for q in AMBIGUOUS_WHITE_STICK_QUERIES:
        rows = _api_fetch_paginated(session, {
            "generator": "search",
            "gsrsearch": q,
            "gsrnamespace": 6,
            "gsrlimit": 30,
        }, max_pages=1)
        for r in rows:
            if r["norm_title"] not in seen_titles and r["norm_title"] not in batch_titles:
                batch_titles.add(r["norm_title"])
                amb_cands.append(r)
        time.sleep(0.15)

    amb_recs: list[dict] = []
    next_a = 1
    for cand in amb_cands:
        if len(amb_recs) >= amb_target:
            break
        out_p = amb_dir / f"v2_wstick_{next_a:03d}.jpg"
        dl = _download_candidate(session, cand, out_p)
        if dl is None:
            continue
        if _is_duplicate(dl["sha256"], dl["norm_title"], dl["ahash"], dl["dhash"], seen_shas, seen_titles, seen_phashes):
            dl["path"].unlink(missing_ok=True)
            continue
        seen_shas.add(dl["sha256"])
        seen_titles.add(dl["norm_title"])
        seen_phashes.append((dl["path"].name, dl["ahash"], dl["dhash"]))
        amb_recs.append({
            "id": f"v2_wstick_{next_a:03d}",
            "abs_path": str(dl["path"]),
            "sha256": dl["sha256"],
            "ahash": dl["ahash"],
            "dhash": dl["dhash"],
            "width": dl["width"],
            "height": dl["height"],
            "title": dl["title"],
            "norm_title": dl["norm_title"],
            "source_cluster": _prefix16(dl["norm_title"]),
            "source_url": dl["page_url"],
            "image_url": dl["image_url"],
            "license": dl["license"],
            "author": dl["author"],
            "description": dl["description"],
        })
        next_a += 1

    payload = {"pos_diverse": pos_recs, "amb_white_sticks": amb_recs}
    meta_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cs_dir = vl_p2_dir / "cache" / "contact_sheets"
    if pos_recs:
        for ch in range(0, len(pos_recs), 48):
            sub = pos_recs[ch:ch + 48]
            make_contact_sheet(
                [(r["id"], load_image_unicode(Path(r["abs_path"]))) for r in sub],
                cs_dir / f"diverse_canes_{ch + 1:02d}_{ch + len(sub):02d}.png",
                cols=6,
            )
    if amb_recs:
        make_contact_sheet(
            [(r["id"], load_image_unicode(Path(r["abs_path"]))) for r in amb_recs],
            cs_dir / "amb_white_sticks.png",
            cols=6,
        )
    print(f"[DiverseCollector] Done: pos_diverse={len(pos_recs)}, amb_white_sticks={len(amb_recs)}", flush=True)
    return payload


if __name__ == "__main__":
    wt = Path(__file__).resolve().parents[3]
    collect_diverse_supplement(wt, pos_target=80, amb_target=18)
