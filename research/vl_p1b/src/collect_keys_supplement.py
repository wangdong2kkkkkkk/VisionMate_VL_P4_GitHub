"""Targeted Wikimedia collector for real modern keys (`pos/object_02`)
and diverse household distractors (mugs, bottles, flashlights, backpacks).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from .collect_vl_p1b_dataset import (
    USER_AGENT,
    _api_fetch,
    _load_vl_p1_signatures,
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

EXTRA_SOURCES = {
    "pos/object_02_keys": {
        "categories": [
            "Category:Pin_tumbler_keys",
            "Category:Dimple_keys",
            "Category:Ignition_keys",
            "Category:Bicycle_lock_keys",
            "Category:Keys_on_key_rings",
            "Category:Transponder_keys",
        ],
        "queries": [
            'intitle:"key" intitle:"ring" -museum -painting -hoyt -print -engraving -opera',
            'intitle:"keys" -piano -keyboard -florida -map -museum -hoyt -opera -locksmith_shop',
            'intitle:"house_key" OR intitle:"house_keys" OR intitle:"car_key" OR intitle:"car_keys"',
            'intitle:"Schluessel" OR intitle:"Schlüssel" -museum -gemälde -zeichnung -relief',
            'intitle:"keychain" keys -lego -usb -toy -doll',
            'intitle:"Abloy" key OR intitle:"Kwikset" key OR intitle:"Schlage" key OR intitle:"Yale_key"',
        ],
        "target": 35,
    },
    "neg/general_household_extra": {
        "categories": [
            "Category:Coffee_mugs",
            "Category:Water_bottles",
            "Category:Flashlights",
            "Category:Backpacks",
        ],
        "queries": [
            'intitle:"coffee_mug" table OR wooden',
            'intitle:"water_bottle" plastic OR metal',
            'intitle:"flashlight" OR intitle:"LED_torch"',
            'intitle:"backpack" studio OR floor -logo',
        ],
        "target": 32,
    },
    "neg/footwear_sock_extra": {
        "categories": [
            "Category:Roller_skates",
            "Category:Ice_skates",
            "Category:Swimfins",
            "Category:Shoe_trees",
        ],
        "queries": [
            'intitle:"roller_skates" OR intitle:"ice_skates" -museum',
            'intitle:"swimfin" OR intitle:"diving_fin" OR intitle:"flippers"',
            'intitle:"insole" OR intitle:"shoe_tree"',
            'intitle:"socks" wool OR striped OR cotton -museum -carape',
        ],
        "target": 24,
    },
    "neg/umbrella_handle_extra": {
        "categories": [
            "Category:Black_umbrellas",
            "Category:Closed_umbrellas",
            "Category:Parasols",
        ],
        "queries": [
            'intitle:"Regenschirm" -museum -print -caricature',
            'intitle:"umbrella" wooden OR handle OR folded OR compact -print -caricature -festival -demonstration',
            'intitle:"kasa" umbrella OR intitle:"wagasa"',
            '"closed umbrella" OR "umbrella handle" OR "umbrellas" rack',
        ],
        "target": 24,
    },
}


def collect_extra(vl_p1_dir: Path, vl_p1b_dir: Path, sleep_sec: float = 0.2) -> dict:
    extra_root = vl_p1b_dir / "data" / "_staging_pool_extra"
    extra_root.mkdir(parents=True, exist_ok=True)
    meta_path = extra_root / "extra_metadata.json"

    seen_shas, seen_titles, seen_phashes = _load_vl_p1_signatures(vl_p1_dir)
    for mp in (
        vl_p1b_dir / "data" / "_staging_pool" / "staging_metadata.json",
        vl_p1b_dir / "data" / "_staging_pool_supplements" / "supplements_metadata.json",
        meta_path,
    ):
        if mp.is_file():
            data = json.loads(mp.read_text(encoding="utf-8"))
            for lst in data.values():
                for rec in lst:
                    seen_shas.add(rec["sha256"])
                    seen_titles.add(rec["norm_title"])
                    seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    existing_meta: dict[str, list[dict]] = {}
    if meta_path.is_file():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for group_key, spec in EXTRA_SOURCES.items():
        group_dir = extra_root / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        recs = [r for r in existing_meta.get(group_key, []) if (extra_root / r["local_rel"]).is_file()]
        target = spec["target"]
        if len(recs) >= target:
            existing_meta[group_key] = recs
            print(f"[Skip extra] {group_key}: {len(recs)}/{target}", flush=True)
            continue

        candidates: list[dict] = []
        seen_batch: set[str] = set()
        for cat in spec.get("categories", []):
            if len(candidates) >= target * 4:
                break
            rows = _api_fetch(session, {
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": 40,
            })
            for row in rows:
                if row["norm_title"] not in seen_batch:
                    seen_batch.add(row["norm_title"])
                    candidates.append(row)
            time.sleep(sleep_sec)

        for q in spec.get("queries", []):
            if len(candidates) >= target * 4:
                break
            rows = _api_fetch(session, {
                "generator": "search",
                "gsrsearch": f"{q} filetype:bitmap",
                "gsrnamespace": 6,
                "gsrlimit": 30,
            })
            for row in rows:
                if row["norm_title"] not in seen_batch:
                    seen_batch.add(row["norm_title"])
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
                sha = hashlib.sha256(resp.content).hexdigest()
                if sha in seen_shas:
                    continue
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None or min(img.shape[:2]) < 150:
                    continue
                ah = compute_ahash(img)
                dh = compute_dhash(img)
                if any(min(hex_hamming_distance(ah, o_ah), hex_hamming_distance(dh, o_dh)) <= 6 for _, o_ah, o_dh in seen_phashes):
                    continue
                fname = f"extra_{idx_counter:03d}.jpg"
                out_p = group_dir / fname
                save_image_unicode(out_p, img)
                rel_p = out_p.relative_to(extra_root).as_posix()
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
        meta_path.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Collected extra] {group_key}: {len(recs)}/{target}", flush=True)

        cs_items = []
        for r in recs:
            p = extra_root / r["local_rel"]
            try:
                cs_items.append((f"{p.stem} | {r['title'][5:]}", load_image_unicode(p), None))
            except ValueError:
                pass
        if cs_items:
            sheet_name = "extra_" + group_key.replace("/", "_") + ".jpg"
            make_contact_sheet(cs_items, vl_p1b_dir / "manifests" / "contact_sheets_v1b" / sheet_name, cols=5)

    return existing_meta


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    collect_extra(root / "research" / "vl_p1", root / "research" / "vl_p1b")
