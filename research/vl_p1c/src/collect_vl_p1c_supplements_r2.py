"""Round-2 Batch 4 collector for extra pos/object_01 (white canes) clusters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from .build_vl_p1c_manifest import _cluster_group_records, _session_cluster_key
from .collect_vl_p1c_dataset import (
    USER_AGENT,
    _api_fetch_paginated,
    _load_existing_signatures,
)
from .common_utils import (
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    load_image_unicode,
    make_contact_sheet,
    save_image_unicode,
)

KEPT_OBJECT_01_STEMS = {
    "cand_015", "cand_041", "cand_042", "cand_043", "cand_044", "cand_045",
    "cand_046", "cand_047", "cand_051", "cand_055", "cand_059",
    "cand_060", "cand_064", "cand_065", "cand_069",
    "cand_077", "cand_082", "cand_085", "cand_086", "cand_089", "cand_091",
    "cand_092", "cand_094", "cand_095", "cand_096", "cand_097", "cand_099",
    "cand_101", "cand_102", "cand_104", "cand_105",
}

QUERIES = [
    "white cane Queen Street Mall",
    "cane tactile paving",
    "vision impaired Brisbane",
    "blind guide dog cane",
    "canne blanche aveugle",
    "Blindenstock Blinder",
    "bastone bianco cieco",
    "taststok blind",
    "white cane walk",
    "blindness white cane",
    "mobilitätstraining blinde langstock",
    "guidedogs white cane",
    "white cane crossing street",
    "international white cane day",
]

TITLE_BAN = (
    "sugar", "candy", "map", "diagram", "1911", "1940", "ca.19", "logo", "icon",
    "festival", "concert", "massacre", "cemetery", "church", "monument", "navcane",
    "mehr,", "bashir", "nuruzzaman", "sharmeen", "shahiduddin", "zahid", "statue",
    "painting", "drawing", "coin", "stamp", "braille",
)


def run_batch4_object01(
    vl_p1_dir: Path,
    vl_p1b_dir: Path,
    vl_p1c_dir: Path,
    target_new: int = 16,
    sleep_sec: float = 0.6,
) -> dict:
    staging = vl_p1c_dir / "data" / "_staging_pool"
    meta_file = staging / "staging_metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    seen_shas, seen_titles, seen_phashes = _load_existing_signatures(vl_p1_dir, vl_p1b_dir, vl_p1c_dir)
    for recs in meta.values():
        for r in recs:
            seen_shas.add(r["sha256"])
            seen_titles.add(r["norm_title"])
            seen_phashes.append((r["local_rel"], r["ahash"], r["dhash"]))

    gkey = "pos/object_01"
    recs = meta[gkey]
    kept_clusters = {
        _session_cluster_key(r["norm_title"])
        for r in recs
        if Path(r["local_rel"]).stem in KEPT_OBJECT_01_STEMS
    }
    print("Current kept clusters:", len(_cluster_group_records(recs, staging, KEPT_OBJECT_01_STEMS)), flush=True)

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})

    cands: dict[str, dict] = {}
    for q in QUERIES:
        time.sleep(0.35)
        rows = _api_fetch_paginated(
            s,
            {"generator": "search", "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 50},
            max_pages=2,
        )
        for r in rows:
            nt = r["norm_title"]
            ck = _session_cluster_key(nt)
            if nt in seen_titles or ck in kept_clusters or ck in cands:
                continue
            tl = r["title"].lower()
            hay = (r["title"] + " " + r["description"]).lower()
            if any(b in tl or b in hay for b in TITLE_BAN):
                continue
            if any(w in hay for w in ("white cane", "long cane", "cane", "canne", "blindenstock", "langstock", "taststok", "bastone")) and (
                any(w in hay for w in ("blind", "vision impaired", "visually impaired", "aveugle", "cieco", "slechtziend", "tactile", "white cane"))
            ):
                cands[ck] = r

    print("Batch 4 unseen candidates:", len(cands), flush=True)
    idx = max([int(Path(r["local_rel"]).stem.split("_")[-1]) for r in recs] + [0]) + 1
    added = []
    for cand in list(cands.values()):
        if len(added) >= target_new:
            break
        resp = None
        for attempt in range(3):
            try:
                r = s.get(cand["image_url"], timeout=25)
                if r.status_code == 429:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                resp = r
                break
            except Exception:
                time.sleep(1.5)
        if resp is None or resp.status_code != 200 or len(resp.content) < 4096:
            time.sleep(sleep_sec)
            continue
        sha_r = hashlib.sha256(resp.content).hexdigest()
        if sha_r in seen_shas:
            continue
        img = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 150:
            continue
        ah, dh = compute_ahash(img), compute_dhash(img)
        if any(hex_hamming_distance(ah, oa) <= 5 and hex_hamming_distance(dh, od) <= 8 for _, oa, od in seen_phashes):
            continue
        out_p = staging / gkey / f"cand_{idx:03d}.jpg"
        save_image_unicode(out_p, img)
        rec = {
            "local_rel": f"{gkey}/cand_{idx:03d}.jpg",
            "group_key": gkey,
            "title": cand["title"],
            "norm_title": cand["norm_title"],
            "source_url": cand["page_url"],
            "image_url": cand["image_url"],
            "license": cand["license"],
            "author": cand["author"],
            "description": cand["description"],
            "sha256": compute_sha256(out_p),
            "ahash": ah,
            "dhash": dh,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
        }
        recs.append(rec)
        added.append(rec)
        seen_shas.add(rec["sha256"])
        seen_shas.add(sha_r)
        seen_titles.add(cand["norm_title"])
        seen_phashes.append((rec["local_rel"], ah, dh))
        idx += 1
        time.sleep(sleep_sec)

    meta[gkey] = recs
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if added:
        items = [
            (Path(r["local_rel"]).stem + " | " + r["title"].replace("File:", "")[:28], load_image_unicode(staging / r["local_rel"]), None)
            for r in added
        ]
        make_contact_sheet(
            items,
            vl_p1c_dir / "manifests" / "contact_sheets_v1c" / "raw_pool_supp_r2" / "pos_object_01_b4.jpg",
            cols=5,
            cell_w=260,
            cell_h=205,
        )
    print(f"[SuppR2-b4] pos/object_01: +{len(added)} -> total {len(recs)}", flush=True)
    return {"added": len(added), "total": len(recs)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run_batch4_object01(root / "research" / "vl_p1", root / "research" / "vl_p1b", root / "research" / "vl_p1c")