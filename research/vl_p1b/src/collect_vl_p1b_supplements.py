"""Supplemental Wikimedia Commons collector for VL-P1b.

Targets high-precision queries for categories where broad category crawling
yielded off-topic photos (e.g., festival portraits for `object_01` white canes,
or empty keyrings for `object_02` bunches of keys), while deduplicating against
both VL-P1 (125 images) and the primary `_staging_pool`.
"""
from __future__ import annotations

import argparse
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

SUPPLEMENT_SOURCES = {
    "pos/object_01": {
        "categories": [
            "Category:Cane_travel",
            "Category:Blindness_aids",
            "Category:White_canes",
        ],
        "queries": [
            '"white cane" pavement OR sidewalk OR street OR stairs -conference -festival -award -speaker',
            '"canne blanche" aveugle OR trottoir',
            '"Blindenstock" OR "Blindenlangstock" OR "Langstock"',
            '"blind" "cane" walking -sugar -candy -speech -podium -microphone',
            'intitle:"white cane" -day -rally -march',
            '"tactile paving" cane',
        ],
        "target": 35,
    },
    "pos/object_02": {
        "categories": [
            "Category:Bunches_of_keys",
            "Category:Latchkeys",
            "Category:House_keys",
            "Category:Car_keys",
        ],
        "queries": [
            '"bunch of keys" -florida -piano -map -keyboard',
            '"set of keys" metal OR brass OR keychain -piano -keyboard -sql -api',
            '"house keys" OR "door keys" metal -realty',
            '"trousseau de clés" OR "trousseau de cles"',
            'intitle:"Schlüssel" OR intitle:"klucze" OR intitle:"chaves" keys',
            '"keys on ring" OR "keys on a ring" OR "key ring with keys"',
        ],
        "target": 35,
    },
    "pos/object_04": {
        "categories": [
            "Category:House_slippers",
            "Category:Flip-flops_(footwear)",
            "Category:Pool_slides",
        ],
        "queries": [
            '"indoor slippers" OR "house slippers"',
            'intitle:"Hausschuhe" -museum -painting -drawing',
            'intitle:"flip-flops" rubber OR foam',
        ],
        "target": 20,
    },
    "neg/glasses_case_confuser": {
        "categories": [
            "Category:Spectacle_cases",
            "Category:Eyeglass_cases",
        ],
        "queries": [
            '"étui à lunettes" OR "etui a lunettes"',
            '"spectacle case" leather OR closed',
            '"eyeglass case" OR "sunglasses case"',
        ],
        "target": 20,
    },
    "neg/notebook_pouch_confuser": {
        "categories": [
            "Category:Moleskine",
            "Category:Leather_notebooks",
            "Category:Pencil_cases",
        ],
        "queries": [
            '"Moleskine" notebook -chart',
            '"leather notebook" OR "leather journal"',
            '"zipper pouch" leather OR canvas',
        ],
        "target": 20,
    },
    "neg/walking_stick_confuser": {
        "categories": [
            "Category:Walking_sticks",
            "Category:Trekking_poles",
            "Category:Crutches",
        ],
        "queries": [
            '"walking stick" wooden -insect -phasmid -bug',
            '"trekking pole" OR "hiking pole" -ski',
            '"crutches" medical OR elbow',
            '"canne de marche" OR "Spazierstock"',
        ],
        "target": 20,
    },
    "neg/long_metal_rod_confuser": {
        "categories": [
            "Category:Ski_poles",
            "Category:Golf_clubs",
            "Category:Crowbars",
            "Category:Monopods",
        ],
        "queries": [
            '"ski poles" aluminum OR pair -portrait',
            '"golf club" iron OR putter',
            '"crowbar" tool metal',
            '"monopod" camera',
        ],
        "target": 22,
    },
    "neg/lock_key_metal_confuser": {
        "categories": [
            "Category:Carabiners",
            "Category:Scissors",
            "Category:Multi-tools",
            "Category:Wrenches",
        ],
        "queries": [
            'intitle:"carabiner" climbing OR metal -key',
            'intitle:"scissors" metal -paper',
            '"pocket knife" OR "multitool" metal -key',
            '"combination padlock" OR "padlock" -key -keys',
            'intitle:"wrench" OR intitle:"pliers" metal',
        ],
        "target": 26,
    },
    "neg/footwear_sock_confuser": {
        "categories": [
            "Category:Argyle_socks",
            "Category:Roller_skates",
            "Category:Ice_skates",
            "Category:Insoles",
            "Category:Swimfins",
        ],
        "queries": [
            'intitle:"socks" pair -carape -museum',
            'intitle:"roller skates" OR intitle:"ice skates"',
            'intitle:"insole" OR "shoe insert" OR "shoe tree"',
            '"tennis socks" OR "wool socks" pair',
        ],
        "target": 22,
    },
    "neg/umbrella_handle_confuser": {
        "categories": [
            "Category:Folded_umbrellas",
            "Category:Umbrella_stands",
            "Category:Compact_umbrellas",
        ],
        "queries": [
            '"folded umbrella" OR "closed umbrella"',
            '"umbrella handle" OR "umbrella stand"',
            'intitle:"Regenschirm" OR intitle:"parapluie" -festival -stage -demonstration',
            '"compact umbrella" OR "folding umbrella"',
        ],
        "target": 22,
    },
}


def collect_supplements(
    vl_p1_dir: Path,
    vl_p1b_dir: Path,
    only_groups: list[str] | None = None,
    sleep_sec: float = 0.2,
) -> dict:
    supp_root = vl_p1b_dir / "data" / "_staging_pool_supplements"
    supp_root.mkdir(parents=True, exist_ok=True)
    supp_meta_path = supp_root / "supplements_metadata.json"

    seen_shas, seen_titles, seen_phashes = _load_vl_p1_signatures(vl_p1_dir)

    # Also deduplicate against primary staging pool if present
    primary_meta_path = vl_p1b_dir / "data" / "_staging_pool" / "staging_metadata.json"
    if primary_meta_path.is_file():
        pm = json.loads(primary_meta_path.read_text(encoding="utf-8"))
        for lst in pm.values():
            for rec in lst:
                seen_shas.add(rec["sha256"])
                seen_titles.add(rec["norm_title"])
                seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    existing_meta: dict[str, list[dict]] = {}
    if supp_meta_path.is_file():
        existing_meta = json.loads(supp_meta_path.read_text(encoding="utf-8"))
        for lst in existing_meta.values():
            for rec in lst:
                seen_shas.add(rec["sha256"])
                seen_titles.add(rec["norm_title"])
                seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    keys = only_groups if only_groups else list(SUPPLEMENT_SOURCES.keys())
    for group_key in keys:
        spec = SUPPLEMENT_SOURCES[group_key]
        group_dir = supp_root / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        recs = [r for r in existing_meta.get(group_key, []) if (supp_root / r["local_rel"]).is_file()]
        target = spec["target"]
        if len(recs) >= target:
            existing_meta[group_key] = recs
            print(f"[Skip supp] {group_key}: {len(recs)}/{target}", flush=True)
            continue

        candidates: list[dict] = []
        seen_batch: set[str] = set()
        for q in spec.get("queries", []):
            if len(candidates) >= target * 4:
                break
            rows = _api_fetch(session, {
                "generator": "search",
                "gsrsearch": f"{q} filetype:bitmap",
                "gsrnamespace": 6,
                "gsrlimit": 40,
            })
            for row in rows:
                if row["norm_title"] not in seen_batch:
                    seen_batch.add(row["norm_title"])
                    candidates.append(row)
            time.sleep(sleep_sec)

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
                fname = f"supp_{idx_counter:03d}.jpg"
                out_p = group_dir / fname
                save_image_unicode(out_p, img)
                rel_p = out_p.relative_to(supp_root).as_posix()
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
        supp_meta_path.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Collected supp] {group_key}: {len(recs)}/{target}", flush=True)

        cs_items = []
        for r in recs:
            p = supp_root / r["local_rel"]
            try:
                cs_items.append((f"{p.stem} | {r['title'][5:]}", load_image_unicode(p), None))
            except ValueError:
                pass
        if cs_items:
            sheet_name = "supp_" + group_key.replace("/", "_") + ".jpg"
            make_contact_sheet(cs_items, vl_p1b_dir / "manifests" / "contact_sheets_v1b" / sheet_name, cols=5)

    return existing_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    collect_supplements(root / "research" / "vl_p1", root / "research" / "vl_p1b", only_groups=args.only)
