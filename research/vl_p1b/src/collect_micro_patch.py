"""Tiny supplemental Wikimedia search-only fetch for modern car/house key bunches
and folded/closed umbrellas with crooked handles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from .collect_vl_p1b_dataset import USER_AGENT, _api_fetch, _load_vl_p1_signatures
from .common_utils import (
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    load_image_unicode,
    make_contact_sheet,
    save_image_unicode,
)

MICRO_SOURCES = {
    "pos/object_02_micro": {
        "categories": [
            "Category:Car_keys",
            "Category:Keychains",
            "Category:Key_fobs",
        ],
        "queries": [
            'intitle:"car_key" OR intitle:"car_keys" OR intitle:"ignition_key" -diagram -logo',
            'intitle:"keychain" OR intitle:"key_fob" -lego -doll -toy',
            'intitle:"keys" ("ring" OR "fob" OR "table" OR "hand") -piano -keyboard -florida -map -hoyt -opera -museum -diagram',
        ],
        "target": 20,
    },
    "pos/object_02_micro2": {
        "categories": [
            "Category:Remote_keyless_systems",
            "Category:Smart_keys",
            "Category:Skeleton_keys",
        ],
        "queries": [
            '"car key" OR "key fob" remote -diagram',
            '"key_ring" OR "keyring" brass OR metal keys -hoyt -opera -museum',
            '"trousseau" cles OR "Schluesselbund" -museum',
        ],
        "target": 15,
    },
    "neg/umbrella_micro": {
        "categories": [
            "Category:Folded_umbrellas",
            "Category:Umbrella_stands",
        ],
        "queries": [
            'intitle:"umbrella_stand" OR intitle:"umbrella_handle"',
            'intitle:"Regenschirm" photo OR jpg -print -museum -caricature',
            '"folded_umbrella" OR "closed_umbrella" -print -caricature',
        ],
        "target": 15,
    },
    "pos/object_03_micro": {
        "categories": [
            "Category:Leather_wallets",
            "Category:Mens_wallets",
            "Category:Bifold_wallets",
        ],
        "queries": [
            'intitle:"leather_wallet" OR intitle:"bifold_wallet" -bitcoin -crypto',
            'intitle:"Portemonnaie" leather OR open -museum -coin',
            '"wallet" pocket OR cash OR leather -bitcoin -crypto -block -hardware',
        ],
        "target": 15,
    },
    "neg/general_household_extra2": {
        "categories": [
            "Category:Water_bottles",
            "Category:Flashlights",
            "Category:Remote_controls",
        ],
        "queries": [
            'intitle:"flashlight" LED -diagram',
            'intitle:"water_bottle" reusable',
            'intitle:"remote_control" tv',
        ],
        "target": 20,
    },
}


def run_micro(vl_p1_dir: Path, vl_p1b_dir: Path, sleep_sec: float = 0.2) -> None:
    extra_root = vl_p1b_dir / "data" / "_staging_pool_extra"
    meta_path = extra_root / "extra_metadata.json"
    existing_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    seen_shas, seen_titles, seen_phashes = _load_vl_p1_signatures(vl_p1_dir)
    for mp in (
        vl_p1b_dir / "data" / "_staging_pool" / "staging_metadata.json",
        vl_p1b_dir / "data" / "_staging_pool_supplements" / "supplements_metadata.json",
        meta_path,
    ):
        if mp.is_file():
            for lst in json.loads(mp.read_text(encoding="utf-8")).values():
                for rec in lst:
                    seen_shas.add(rec["sha256"])
                    seen_titles.add(rec["norm_title"])
                    seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for group_key, spec in MICRO_SOURCES.items():
        group_dir = extra_root / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        recs = [r for r in existing_meta.get(group_key, []) if (extra_root / r["local_rel"]).is_file()]
        target = spec["target"]
        candidates, seen_b = [], set()
        for cat in spec["categories"]:
            for row in _api_fetch(session, {"generator": "categorymembers", "gcmtitle": cat, "gcmtype": "file", "gcmlimit": 25}):
                if row["norm_title"] not in seen_b:
                    seen_b.add(row["norm_title"])
                    candidates.append(row)
            time.sleep(sleep_sec)
        for q in spec["queries"]:
            for row in _api_fetch(session, {"generator": "search", "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 25}):
                if row["norm_title"] not in seen_b:
                    seen_b.add(row["norm_title"])
                    candidates.append(row)
            time.sleep(sleep_sec)

        idx_counter = len(recs) + 1
        for cand in candidates:
            if len(recs) >= target:
                break
            if cand["norm_title"] in seen_titles or not cand["image_url"]:
                continue
            try:
                resp = session.get(cand["image_url"], timeout=25)
                if resp.status_code != 200 or len(resp.content) < 4096:
                    continue
                sha = hashlib.sha256(resp.content).hexdigest()
                if sha in seen_shas:
                    continue
                img = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None or min(img.shape[:2]) < 150:
                    continue
                ah, dh = compute_ahash(img), compute_dhash(img)
                if any(min(hex_hamming_distance(ah, oa), hex_hamming_distance(dh, od)) <= 6 for _, oa, od in seen_phashes):
                    continue
                out_p = group_dir / f"micro_{idx_counter:03d}.jpg"
                save_image_unicode(out_p, img)
                rel_p = out_p.relative_to(extra_root).as_posix()
                rec = {
                    "local_rel": rel_p,
                    "group_key": group_key,
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
        meta_path.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Collected micro] {group_key}: {len(recs)}/{target}", flush=True)
        cs_items = [(f"{(extra_root / r['local_rel']).stem} | {r['title'][5:]}", load_image_unicode(extra_root / r["local_rel"]), None) for r in recs]
        if cs_items:
            make_contact_sheet(cs_items, vl_p1b_dir / "manifests" / "contact_sheets_v1b" / ("micro_" + group_key.replace("/", "_") + ".jpg"), cols=5)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run_micro(root / "research" / "vl_p1", root / "research" / "vl_p1b")
