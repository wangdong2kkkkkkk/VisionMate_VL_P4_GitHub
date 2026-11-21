"""Step 3a-supp: Targeted Wikimedia Commons supplement collector for groups needing cleaner/more diverse clusters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests

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

SUPP_SPECS: dict[str, dict] = {
    "pos/object_01": {
        "extra": 55,
        "ban_words": (
            "festival", "safety day", "conference", "award", "podium", "concert",
            "guitar", "orchestra", "goalball", "minton", "bell", "monument",
            "miniature", "manuscript", "comic", "cartoon", "drawing", "map",
            "plaque", "disc golf", "mower", "trombone", "trumpet", "violin",
        ),
        "categories": [
            "Category:White_canes",
            "Category:Blind_people_with_white_canes",
            "Category:Guide_dogs_and_white_canes",
        ],
        "queries": [
            '"white cane" walking OR pavement OR blind -festival -conference -award -goalball -minton -safety',
            '"canne blanche" aveugle OR marchant -festival -ceremony',
            '"Blindenstock" OR "Blindenlangstock" OR "Langstock" Straße OR Weg',
            '"bastón blanco" OR "baston blanco" ciego OR calle',
            '"folded cane" OR "folding cane" OR "mobility cane" white',
            '"tactile paving" "cane" blind OR pedestrian',
            '"白杖" 視覚障害 OR 点字ブロック',
            '"белая трость" незрячий OR слепой',
            '"biała laska" niewidomy',
            '"blind person" cane street OR crosswalk OR subway OR train',
        ],
    },
    "pos/object_02": {
        "extra": 35,
        "ban_words": ("hydrant", "findid", "roman", "medieval", "plush", "remove before flight", "alcatraz", "patent"),
        "categories": [
            "Category:Keys",
            "Category:House_keys",
            "Category:Bunches_of_keys",
            "Category:Keys_on_key_rings",
            "Category:Pin_tumbler_keys",
            "Category:Car_keys",
        ],
        "queries": [
            'intitle:"bunch of keys" OR intitle:"set of keys" -diagram -museum',
            'intitle:"house key" OR intitle:"door key" metal',
            'intitle:"car key" remote OR transponder',
            'intitle:"Schlüssel" Schlüsselbund OR Haustür -museum -castle',
            'intitle:"trousseau" clés OR porte',
            'intitle:"chiavi" OR intitle:"llaves" metal',
        ],
    },
    "pos/object_03": {
        "extra": 30,
        "ban_words": ("capa-caderno", "capa-agenda", "jogo-escritorio", "bitcoin", "crypto", "tablet"),
        "categories": [
            "Category:Bifold_wallets",
            "Category:Mens_wallets",
            "Category:Leather_wallets",
            "Category:Billfolds",
            "Category:Coin_purses",
        ],
        "queries": [
            'intitle:"leather wallet" OR intitle:"bifold wallet" -crypto -bitcoin',
            'intitle:"wallet" cash OR credit card -digital -software',
            'intitle:"Portemonnaie" Leder OR Euro',
            'intitle:"coin purse" leather OR zipper',
        ],
    },
    "neg/glasses_case_confuser": {
        "extra": 25,
        "ban_words": ("mussel", "microscope", "ivory", "cumberlandia", "senator", "bundesarchiv", "bow,"),
        "categories": ["Category:Spectacle_cases", "Category:Eyeglass_cases"],
        "queries": [
            '"glasses case" OR "eyeglass case" OR "sunglasses case" -mussel',
            '"Brillenetui" OR "étui à lunettes" OR "spectacle case" -museum',
            '"zipper" "glasses case" OR "hard glasses case"',
        ],
    },
    "neg/notebook_pouch_confuser": {
        "extra": 25,
        "ban_words": ("cuaderno de notas de 18", "fatim", "fisa de instruire", "nithyananda", "laptop"),
        "categories": ["Category:Moleskine", "Category:Spiral_notebooks", "Category:Pencil_cases", "Category:Notebooks"],
        "queries": [
            'intitle:"Moleskine" notebook -portrait -drawing',
            'intitle:"spiral notebook" OR intitle:"leather journal" desk',
            'intitle:"passport cover" OR intitle:"passport holder" leather',
            'intitle:"zipper pouch" OR intitle:"pencil case" fabric OR leather',
        ],
    },
    "neg/walking_stick_confuser": {
        "extra": 25,
        "ban_words": ("vihorlat", "museum", "indian election", "drawing", "stool", "report on surgical", "vrata"),
        "categories": ["Category:Walking_sticks", "Category:Trekking_poles", "Category:Crutches"],
        "queries": [
            'intitle:"walking stick" wood OR handle -insect -phasmid -museum',
            'intitle:"trekking poles" OR intitle:"hiking poles" pair',
            'intitle:"Spazierstock" Holz OR Griff',
            'intitle:"forearm crutch" OR intitle:"crutch" aluminum',
        ],
    },
    "neg/long_metal_rod_confuser": {
        "extra": 40,
        "ban_words": ("golf club", "ko olina", "palace", "scotland fife", "fountain", "pride", "street f", "conference", "graffiti", "amujeo"),
        "categories": ["Category:Crowbars", "Category:Selfie_sticks", "Category:Monopods", "Category:Pool_cues", "Category:Brooms"],
        "queries": [
            'intitle:"crowbar" tool metal',
            'intitle:"selfie stick" monopod OR extendable -festival -parade -pride',
            'intitle:"billiard cue" OR intitle:"pool cue" wooden OR stick',
            'intitle:"golf iron" OR intitle:"putter" golf head -course -clubhouse',
            'intitle:"tripod" aluminum OR metal studio',
            'intitle:"microphone stand" OR intitle:"fishing rod" closeup OR reel',
        ],
    },
    "neg/footwear_sock_confuser": {
        "extra": 35,
        "ban_words": ("carape", "opanci", "dagestan", "marcha orgulho", "ferdig", "plushev", "shoe"),
        "categories": ["Category:Socks", "Category:Argyle_socks", "Category:Wool_socks", "Category:Insoles", "Category:Shoe_trees"],
        "queries": [
            'intitle:"wool socks" OR intitle:"knit socks" OR intitle:"striped socks" pair',
            'intitle:"argyle socks" OR intitle:"ankle socks" OR intitle:"crew socks"',
            'intitle:"Socken" Wolle OR Paar',
            'intitle:"shoe tree" wood OR cedar',
            'intitle:"insole" OR intitle:"shoehorn" leather OR foam',
        ],
    },
    "neg/umbrella_handle_confuser": {
        "extra": 30,
        "ban_words": ("apaph", "baseball", "parachut", "brenn", "ragsdale", "altro", "monument", "2014-035"),
        "categories": ["Category:Folded_umbrellas", "Category:Closed_umbrellas", "Category:Black_umbrellas", "Category:Umbrella_stands"],
        "queries": [
            'intitle:"folded umbrella" OR intitle:"closed umbrella" OR intitle:"compact umbrella"',
            'intitle:"umbrella handle" OR intitle:"black umbrella" closed OR rain',
            'intitle:"Regenschirm" Griff OR zugeklappt',
            'intitle:"parapluie" fermé OR poignée',
        ],
    },
    "neg/general_distractor": {
        "extra": 45,
        "ban_words": ("actress", "festival", "poster", "portrait"),
        "categories": [
            "Category:Backpacks",
            "Category:Water_bottles",
            "Category:Flashlights",
            "Category:Headphones",
            "Category:Eyeglasses",
            "Category:Remote_controls",
        ],
        "queries": [
            'intitle:"backpack" travel OR hiking',
            'intitle:"flashlight" LED OR tactical',
            'intitle:"headphones" studio OR wireless',
            'intitle:"water bottle" OR intitle:"thermos" stainless',
            'intitle:"eyeglasses" frame OR desk',
        ],
    },
    "amb/ambiguous_boundary": {
        "extra": 12,
        "ban_words": ("baycityrollers", "festival", "painting", "1887", "band"),
        "categories": ["Category:Antique_keys", "Category:Wooden_clogs"],
        "queries": [
            'intitle:"skeleton key" OR intitle:"antique key" iron OR brass',
            'intitle:"clogs" OR intitle:"klompen" pair leather OR wooden',
            'intitle:"crocs" clog shoes',
        ],
    },
}


def run_supplements(vl_p1_dir: Path, vl_p1b_dir: Path, vl_p1c_dir: Path, sleep_sec: float = 0.12) -> dict:
    staging_root = vl_p1c_dir / "data" / "_staging_pool"
    meta_file = staging_root / "staging_metadata.json"
    meta: dict[str, list[dict]] = json.loads(meta_file.read_text(encoding="utf-8"))

    seen_shas, seen_titles, seen_phashes = _load_existing_signatures(vl_p1_dir, vl_p1b_dir)
    for lst in meta.values():
        for rec in lst:
            seen_shas.add(rec["sha256"])
            seen_titles.add(rec["norm_title"])
            seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    cs_dir = vl_p1c_dir / "manifests" / "contact_sheets_v1c" / "raw_pool_supp"
    cs_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    orig_counts = {
        "pos/object_01": 58, "pos/object_02": 58, "pos/object_03": 58,
        "neg/glasses_case_confuser": 38, "neg/notebook_pouch_confuser": 38,
        "neg/walking_stick_confuser": 38, "neg/long_metal_rod_confuser": 38,
        "neg/footwear_sock_confuser": 38, "neg/umbrella_handle_confuser": 38,
        "neg/general_distractor": 175, "amb/ambiguous_boundary": 18,
    }
    for gkey, spec in SUPP_SPECS.items():
        gdir = staging_root / gkey
        gdir.mkdir(parents=True, exist_ok=True)
        recs = meta.get(gkey, [])
        base_n = orig_counts[gkey]
        ban = tuple(w.lower() for w in spec.get("ban_words", ()))
        extra = int(spec["extra"])

        if len(recs) >= base_n + extra:
            added_recs = recs[base_n:]
        else:
            # Gather diverse candidates across ALL categories AND queries round-robin
            cand_pools: list[list[dict]] = []
            seen_batch: set[str] = set()
            for cat in spec.get("categories", []):
                rows = _api_fetch_paginated(session, {
                    "generator": "categorymembers",
                    "gcmtitle": cat,
                    "gcmtype": "file",
                    "gcmlimit": 50,
                }, max_pages=3)
                cand_pools.append(rows)
                time.sleep(sleep_sec)
            for q in spec.get("queries", []):
                rows = _api_fetch_paginated(session, {
                    "generator": "search",
                    "gsrsearch": f"{q} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                }, max_pages=2)
                cand_pools.append(rows)
                time.sleep(sleep_sec)

            # Interleave round-robin so one noisy category doesn't starve queries!
            cands: list[dict] = []
            max_len = max((len(p) for p in cand_pools), default=0)
            for row_i in range(max_len):
                for pool in cand_pools:
                    if row_i >= len(pool):
                        continue
                    row = pool[row_i]
                    nt = row["norm_title"]
                    if nt in seen_batch or nt in seen_titles:
                        continue
                    hay = f"{row['title']} {row['description']}".lower()
                    if any(b in hay for b in ban):
                        continue
                    seen_batch.add(nt)
                    cands.append(row)

            idx_counter = max([int(Path(r["local_rel"]).stem.split("_")[-1]) for r in recs] + [0]) + 1
            added_recs = []

            for cand in cands:
                if len(recs) >= base_n + extra:
                    break
                url = cand.get("image_url")
                if not url:
                    continue
                try:
                    resp = session.get(url, timeout=25)
                    if resp.status_code != 200 or len(resp.content) < 4096:
                        time.sleep(sleep_sec)
                        continue
                    sha_raw = hashlib.sha256(resp.content).hexdigest()
                    if sha_raw in seen_shas:
                        continue
                    img = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None or min(img.shape[:2]) < 150:
                        continue
                    ah, dh = compute_ahash(img), compute_dhash(img)
                    if any(min(hex_hamming_distance(ah, oa), hex_hamming_distance(dh, od)) <= 6 for _, oa, od in seen_phashes):
                        continue

                    fname = f"cand_{idx_counter:03d}.jpg"
                    out_p = gdir / fname
                    save_image_unicode(out_p, img)
                    rel_p = out_p.relative_to(staging_root).as_posix()
                    rec = {
                        "local_rel": rel_p,
                        "group_key": gkey,
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
                    }
                    recs.append(rec)
                    added_recs.append(rec)
                    seen_shas.add(rec["sha256"])
                    seen_shas.add(sha_raw)
                    seen_titles.add(cand["norm_title"])
                    seen_phashes.append((rel_p, ah, dh))
                    idx_counter += 1
                    time.sleep(sleep_sec)
                except Exception:
                    time.sleep(sleep_sec)

            meta[gkey] = recs
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        if added_recs:
            sheet_items = []
            for r in added_recs:
                p = staging_root / r["local_rel"]
                try:
                    sheet_items.append((f"{p.stem} | {r['title'].replace('File:', '')[:28]}", load_image_unicode(p), None))
                except ValueError:
                    pass
            safe = gkey.replace("/", "_")
            for p_idx in range(0, len(sheet_items), 30):
                suf = f"_p{p_idx // 30 + 1}" if len(sheet_items) > 30 else ""
                make_contact_sheet(sheet_items[p_idx:p_idx + 30], cs_dir / f"{safe}{suf}.jpg", cols=5, cell_w=260, cell_h=205)

        summary[gkey] = {"added": len(added_recs), "total": len(recs)}
        print(f"[Supp] {gkey}: +{len(added_recs)} -> total {len(recs)}", flush=True)

    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run_supplements(root / "research" / "vl_p1", root / "research" / "vl_p1b", root / "research" / "vl_p1c")
