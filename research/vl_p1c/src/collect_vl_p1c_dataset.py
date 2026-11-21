"""Step 3a: Collect fresh open-licensed Wikimedia Commons photos for VL-P1c.

Guarantees zero leakage against BOTH VL-P1 (125 images) and VL-P1b (train/gallery/cal/holdout/ambiguous)
across SHA-256, normalized title, session cluster, and perceptual aHash/dHash <= 6.

Targets (with visual curation buffer):
- 5 positive classes: 30 new needed per class (10 new cal_v1c + 20 new final_holdout_v1c) -> fetch ~55 candidates/class
- 7 hard-negative groups: 24-26 new needed per group (8-9 cal_v1c + 16 final_holdout_v1c) -> fetch ~38 candidates/group
- general_distractor: 129 new needed (41 cal_v1c + 88 final_holdout_v1c) -> fetch ~175 candidates
- ambiguous_boundary: 6+ new needed -> fetch ~16 candidates
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import requests

from .common_utils import (
    compute_ahash,
    compute_dhash,
    compute_sha256,
    hex_hamming_distance,
    load_image_unicode,
    make_contact_sheet,
    save_image_unicode,
)
from .labeling_protocol import is_allowed_open_license

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VisionMateVLP1cResearch/1.0 (https://github.com/visionmate-research; academic benchmark)"

EXCLUDE_TITLE_WORDS = (
    "logo", "icon", "diagram", "chart", "map", "flag", "coat of arms", "seal of",
    "poster", "advertisement", "newspaper", "cartoon", "drawing", "sketch", "painting",
    "illustration", "stamp", "coin", "medal", "blueprint", "patent", "svg", "score",
    "sheet music", "herbarium", "fossil", "mushroom", "butterfly", "beetle", "fish",
    "orchid", "candy", "sugar cane", "sugarcane", "bitcoin", "crypto", "screenshot",
    "florida keys", "keyboard", "piano", "typewriter", "organ", "saxophone", "flute",
    "horseshoe", "brake shoe", "snowshoe", "statue", "sculpture", "engraving", "caricature",
    "rally", "conference", "award", "podium", "opera", "hoyt",
)

POSITIVE_SOURCES = {
    "object_01": {
        "categories": [
            "Category:White_canes",
            "Category:Blind_people_with_white_canes",
            "Category:Cane_travel",
            "Category:Blindness_aids",
            "Category:International_White_Cane_Safety_Day",
            "Category:Orientation_and_mobility",
            "Category:Guide_dogs_and_white_canes",
        ],
        "queries": [
            '"white cane" blind -sugar -candy -speech -award',
            '"long cane" mobility blind',
            '"canne blanche" aveugle OR trottoir',
            '"Blindenstock" OR "Blindenlangstock" OR "Langstock"',
            '"Bastón blanco" OR "baston blanco" ciego',
            '"vit käpp" OR "blinndenstock" OR "blindenstok"',
            'intitle:"white cane" -conference -award -rally',
            '"tactile paving" white cane OR blind',
            '"blind" "cane" walking pavement OR sidewalk OR stairs',
            '"mobility cane" OR "folding cane" blind',
        ],
        "target": 58,
    },
    "object_02": {
        "categories": [
            "Category:Key_rings",
            "Category:Keys",
            "Category:Bunches_of_keys",
            "Category:House_keys",
            "Category:Car_keys",
            "Category:Keys_on_key_rings",
            "Category:Keychains",
            "Category:Key_fobs",
            "Category:Pin_tumbler_keys",
            "Category:Dimple_keys",
        ],
        "queries": [
            'intitle:"bunch of keys" -florida -piano -map',
            'intitle:"keychain" OR intitle:"keyring" keys -usb -lego -doll',
            'intitle:"Schlüsselbund" OR intitle:"Schluesselbund" -museum',
            'intitle:"car_keys" OR intitle:"house_keys" OR intitle:"house_key"',
            '"trousseau de clés" OR "trousseau de cles"',
            'intitle:"klucze" OR intitle:"chaves" OR intitle:"sleutels" keys',
            '"keys on a ring" OR "set of keys" metal -piano -keyboard -sql',
            '"Yale_key" OR "Abloy" key OR "Kwikset" key ring',
            '"car key" fob remote -diagram',
        ],
        "target": 58,
    },
    "object_03": {
        "categories": [
            "Category:Wallets",
            "Category:Leather_wallets",
            "Category:Mens_wallets",
            "Category:Bifold_wallets",
            "Category:Coin_purses",
            "Category:Billfolds",
        ],
        "queries": [
            'intitle:"wallet" leather -bitcoin -crypto -hardware',
            'intitle:"Portemonnaie" OR intitle:"Portefeuille" -castle -coin',
            'intitle:"bifold" OR intitle:"billfold" wallet',
            'intitle:"coin purse" -crochet -diagram',
            '"leather wallet" pocket OR cash OR cards',
            '"trifold wallet" OR "zip wallet" leather',
            'intitle:"portafoglio" OR intitle:"cartera" wallet',
            'intitle:"Geldbörse" OR intitle:"Brieftasche"',
        ],
        "target": 58,
    },
    "object_04": {
        "categories": [
            "Category:Flip-flops_(footwear)",
            "Category:Slippers",
            "Category:House_slippers",
            "Category:Sandals",
            "Category:Pool_slides",
            "Category:Zori",
            "Category:Slides_(footwear)",
        ],
        "queries": [
            'intitle:"flip-flops" OR intitle:"flip flops" OR intitle:"thong_sandals"',
            'intitle:"slippers" -orchid -lobster -fish -museum',
            'intitle:"Hausschuhe" OR intitle:"Pantoffel" OR intitle:"Latschen"',
            'intitle:"sandals" beach OR leather OR pair -statue',
            '"house slippers" OR "indoor slippers" pair',
            '"slide sandals" OR "pool slides" OR "rubber sandals"',
            'intitle:"chinelos" OR intitle:"tongs" sandals',
            'intitle:"geta" sandals OR "zori" sandals',
        ],
        "target": 58,
    },
    "object_05": {
        "categories": [
            "Category:Sneakers",
            "Category:Running_shoes",
            "Category:Athletic_shoes",
            "Category:Basketball_shoes",
            "Category:Tennis_shoes",
            "Category:Oxford_shoes",
            "Category:Dress_shoes",
            "Category:Leather_shoes",
            "Category:Loafers",
        ],
        "queries": [
            'intitle:"sneakers" -building -store',
            'intitle:"running_shoes" OR intitle:"athletic_shoes" OR intitle:"trainer" shoes',
            'intitle:"Turnschuhe" OR intitle:"Sportschuhe" OR intitle:"Laufschuhe"',
            'intitle:"leather_shoes" OR intitle:"oxford_shoes" OR intitle:"brogues"',
            'intitle:"hiking_boots" OR intitle:"work_boots" pair',
            '"basketball shoes" OR "tennis shoes" pair',
            'intitle:"converse" OR intitle:"adidas" OR intitle:"nike" shoes',
            'intitle:"chaussures" sport OR cuir',
        ],
        "target": 58,
    },
}

NEGATIVE_SOURCES = {
    "glasses_case_confuser": {
        "categories": [
            "Category:Spectacle_cases",
            "Category:Eyeglass_cases",
            "Category:Pencil_cases",
        ],
        "queries": [
            '"glasses case" OR "eyeglass case" OR "spectacle case"',
            '"sunglasses case" OR "Brillenetui" OR "étui à lunettes"',
            '"hard glasses case" OR "leather spectacle case"',
            '"pencil case" leather OR zipper',
        ],
        "target": 38,
    },
    "notebook_pouch_confuser": {
        "categories": [
            "Category:Notebooks",
            "Category:Moleskine",
            "Category:Leather_notebooks",
            "Category:Leather_bindings",
            "Category:Passport_holders",
        ],
        "queries": [
            'intitle:"notebook" leather OR pocket OR hardcover',
            'intitle:"Moleskine" OR intitle:"Field Notes" notebook',
            'intitle:"Notizbuch" OR intitle:"carnet" leather',
            '"leather journal" OR "zipper pouch" OR "passport cover"',
            '"clutch bag" leather OR "cosmetic pouch"',
        ],
        "target": 38,
    },
    "walking_stick_confuser": {
        "categories": [
            "Category:Walking_sticks",
            "Category:Trekking_poles",
            "Category:Crutches",
            "Category:Canes",
            "Category:Ski_poles",
        ],
        "queries": [
            'intitle:"walking_stick" wooden OR carved -insect -phasmid',
            'intitle:"trekking_pole" OR intitle:"hiking_pole" OR "nordic walking"',
            'intitle:"Spazierstock" OR intitle:"Wanderstock" OR "canne de marche"',
            'intitle:"crutches" OR "forearm crutch" -cartoon',
            'intitle:"shepherd_crook" OR "irish blackthorn" stick',
        ],
        "target": 38,
    },
    "long_metal_rod_confuser": {
        "categories": [
            "Category:Selfie_sticks",
            "Category:Golf_clubs",
            "Category:Crowbars",
            "Category:Monopods",
            "Category:Tripods",
            "Category:Fishing_rods",
            "Category:Pool_cues",
        ],
        "queries": [
            'intitle:"selfie_stick" OR intitle:"monopod" camera',
            'intitle:"crowbar" OR intitle:"pry_bar" metal',
            'intitle:"golf_club" OR "golf_iron" OR "putter" shaft',
            'intitle:"ski_poles" aluminum OR pair -portrait',
            'intitle:"tripod_leg" OR "curtain_rod" OR "metal_rod"',
            '"pool cue" OR "billiard cue" OR "boathook" OR "broom handle"',
        ],
        "target": 38,
    },
    "lock_key_metal_confuser": {
        "categories": [
            "Category:Padlocks",
            "Category:Carabiners",
            "Category:Scissors",
            "Category:Spoons",
            "Category:Wrenches",
            "Category:Multi-tools",
            "Category:USB_flash_drives",
        ],
        "queries": [
            'intitle:"padlock" brass OR steel -key',
            'intitle:"carabiner" climbing OR metal -key',
            'intitle:"scissors" metal OR stainless',
            'intitle:"wrench" OR intitle:"spanner" OR intitle:"pliers" tool',
            'intitle:"pocket_knife" OR intitle:"multitool" metal',
            'intitle:"spoons" cutlery OR "fork" stainless',
            'intitle:"usb_flash_drive" metal OR keychain -key',
        ],
        "target": 40,
    },
    "footwear_sock_confuser": {
        "categories": [
            "Category:Socks",
            "Category:Argyle_socks",
            "Category:Wool_socks",
            "Category:Roller_skates",
            "Category:Ice_skates",
            "Category:Swimfins",
            "Category:Insoles",
            "Category:Geta",
            "Category:Shoe_trees",
        ],
        "queries": [
            'intitle:"socks" wool OR cotton OR striped -museum',
            'intitle:"Socken" OR "chaussettes" pair',
            'intitle:"roller_skates" OR intitle:"ice_skates" -museum',
            'intitle:"swimfins" OR intitle:"diving_fins" OR "insole"',
            'intitle:"shoe_cover" OR "boot_cover" OR "leg_warmers"',
            'intitle:"snowshoes" OR "shoe_tree" wooden',
        ],
        "target": 38,
    },
    "umbrella_handle_confuser": {
        "categories": [
            "Category:Umbrellas",
            "Category:Folded_umbrellas",
            "Category:Closed_umbrellas",
            "Category:Black_umbrellas",
            "Category:Compact_umbrellas",
            "Category:Umbrella_stands",
        ],
        "queries": [
            'intitle:"umbrella" folded OR closed OR handle -festival -demonstration',
            'intitle:"Regenschirm" OR intitle:"parapluie" -print -caricature -stage',
            '"compact umbrella" OR "umbrella handle" OR "umbrellas stand"',
            'intitle:"kasa" OR intitle:"wagasa" umbrella',
        ],
        "target": 38,
    },
    "general_distractor": {
        "categories": [
            "Category:Wristwatches",
            "Category:Computer_mice",
            "Category:Coffee_mugs",
            "Category:Water_bottles",
            "Category:Flashlights",
            "Category:Eyeglasses",
            "Category:Mobile_phones",
            "Category:Backpacks",
            "Category:Remote_controls",
            "Category:Headphones",
            "Category:Chairs",
            "Category:Bicycles",
            "Category:Handbags",
            "Category:Camera_lenses",
            "Category:Staircases",
            "Category:Handrails",
        ],
        "queries": [
            'intitle:"wristwatch" OR intitle:"coffee_mug" table',
            'intitle:"computer_mouse" OR intitle:"smartphone" desk',
            'intitle:"flashlight" OR intitle:"water_bottle" OR "thermos"',
            'intitle:"backpack" OR intitle:"headphones" studio',
            'intitle:"eyeglasses" OR intitle:"sunglasses" table',
            'intitle:"remote_control" OR intitle:"calculator" OR "alarm_clock"',
            'intitle:"handrail" OR intitle:"staircase" OR "sidewalk"',
            'intitle:"stapler" OR intitle:"book" OR "tea_cup" wooden',
            'intitle:"chair" wooden OR "dining_table" OR "desk_lamp"',
            'intitle:"bicycle" street OR "keyboard_musical" -key',
        ],
        "target": 175,
    },
}

AMBIGUOUS_SOURCES = {
    "ambiguous_boundary": {
        "categories": [
            "Category:Clogs",
            "Category:Antique_keys",
            "Category:Crocs",
            "Category:Sandals",
        ],
        "queries": [
            'intitle:"clogs" OR intitle:"wooden_clogs" OR "sabots" pair',
            'intitle:"antique_key" OR intitle:"skeleton_key" single',
            'intitle:"crocs" OR "garden_clogs" footwear',
            '"utility pouch" OR "wristlet" leather',
        ],
        "target": 18,
    }
}


def _strip_html(raw: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", raw or "")
    return " ".join(txt.split())


def _normalize_title(title: str) -> str:
    t = title.strip().lower()
    if t.startswith("file:"):
        t = t[5:]
    return re.sub(r"[\s_]+", "_", t)


def _is_clean_title(title: str) -> bool:
    tl = title.lower()
    if not tl.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return False
    return not any(bad in tl for bad in EXCLUDE_TITLE_WORDS)


def _load_existing_signatures(*research_dirs: Path) -> tuple[set[str], set[str], list[tuple[str, str, str]]]:
    """Load SHA-256, normalized titles/URLs, and aHash/dHash from VL-P1 & VL-P1b manifests + files."""
    seen_shas: set[str] = set()
    seen_titles: set[str] = set()
    seen_phashes: list[tuple[str, str, str]] = []

    for rdir in research_dirs:
        mf_p = rdir / "manifests" / "dataset_manifest.json"
        if mf_p.is_file():
            mf = json.loads(mf_p.read_text(encoding="utf-8"))
            # VL-P1 schema uses train/calibration/test; VL-P1b schema uses train_pos/hard_neg_gallery/...
            for key, val in mf.items():
                if not isinstance(val, list):
                    continue
                for item in val:
                    if not isinstance(item, dict):
                        continue
                    for sk in ("sha256", "full_sha256", "roi_sha256"):
                        if item.get(sk):
                            seen_shas.add(item[sk])
                    if item.get("title"):
                        seen_titles.add(_normalize_title(item["title"]))
                    if item.get("source_url"):
                        last = item["source_url"].rstrip("/").split("/")[-1]
                        seen_titles.add(_normalize_title(last))
                    if item.get("ahash") and item.get("dhash"):
                        seen_phashes.append((str(item.get("id", key)), item["ahash"], item["dhash"]))
                    if item.get("full_ahash") and item.get("full_dhash"):
                        seen_phashes.append((str(item.get("id", key)) + "_full", item["full_ahash"], item["full_dhash"]))

        data_root = rdir / "data"
        if data_root.exists():
            for p in data_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    if "_staging" in str(p):
                        continue
                    seen_shas.add(compute_sha256(p))
                    try:
                        img = load_image_unicode(p)
                        seen_phashes.append((p.name, compute_ahash(img), compute_dhash(img)))
                    except ValueError:
                        pass

    return seen_shas, seen_titles, seen_phashes


def _api_fetch_paginated(session: requests.Session, params: dict, max_pages: int = 2) -> list[dict]:
    out: list[dict] = []
    cur_params = dict(params)
    for _ in range(max_pages):
        base_params = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
        }
        base_params.update(cur_params)
        data = None
        for attempt in range(3):
            try:
                r = session.get(COMMONS_API, params=base_params, timeout=25)
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
            if min(w, h) < 160 or max(w, h) > 3200:
                continue
            meta = ii.get("extmetadata", {})
            lic = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not is_allowed_open_license(lic):
                continue
            out.append({
                "title": title,
                "norm_title": _normalize_title(title),
                "page_url": ii.get("descriptionurl", ""),
                "image_url": ii.get("thumburl") or ii.get("url"),
                "orig_url": ii.get("url", ""),
                "license": lic,
                "author": _strip_html(meta.get("Artist", {}).get("value", "Unknown"))[:120],
                "description": _strip_html(meta.get("ImageDescription", {}).get("value", ""))[:220],
            })
        cont = data.get("continue")
        if not cont:
            break
        cur_params.update(cont)
    return out


def collect_staging_pool_v1c(
    vl_p1_dir: Path,
    vl_p1b_dir: Path,
    vl_p1c_dir: Path,
    only_groups: list[str] | None = None,
    sleep_sec: float = 0.15,
) -> dict:
    staging_root = vl_p1c_dir / "data" / "_staging_pool"
    staging_root.mkdir(parents=True, exist_ok=True)
    meta_file = staging_root / "staging_metadata.json"

    seen_shas, seen_titles, seen_phashes = _load_existing_signatures(vl_p1_dir, vl_p1b_dir)
    existing_meta: dict[str, list[dict]] = {}
    if meta_file.is_file():
        existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for lst in existing_meta.values():
            for rec in lst:
                seen_shas.add(rec["sha256"])
                seen_titles.add(rec["norm_title"])
                seen_phashes.append((rec["local_rel"], rec["ahash"], rec["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_specs = {}
    all_specs.update({f"pos/{k}": v for k, v in POSITIVE_SOURCES.items()})
    all_specs.update({f"neg/{k}": v for k, v in NEGATIVE_SOURCES.items()})
    all_specs.update({f"amb/{k}": v for k, v in AMBIGUOUS_SOURCES.items()})

    keys = only_groups if only_groups else list(all_specs.keys())
    for group_key in keys:
        spec = all_specs[group_key]
        group_dir = staging_root / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        recs = [r for r in existing_meta.get(group_key, []) if (staging_root / r["local_rel"]).is_file()]
        target = spec["target"]
        if len(recs) >= target:
            existing_meta[group_key] = recs
            print(f"[Skip] {group_key}: already has {len(recs)}/{target}", flush=True)
            continue

        candidates: list[dict] = []
        seen_batch_titles: set[str] = set()
        for cat in spec.get("categories", []):
            if len(candidates) >= target * 4:
                break
            rows = _api_fetch_paginated(session, {
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": 50,
            }, max_pages=2)
            for row in rows:
                if row["norm_title"] not in seen_batch_titles and row["norm_title"] not in seen_titles:
                    seen_batch_titles.add(row["norm_title"])
                    candidates.append(row)
            time.sleep(sleep_sec)

        if len(candidates) < target * 3:
            for q in spec.get("queries", []):
                if len(candidates) >= target * 4:
                    break
                rows = _api_fetch_paginated(session, {
                    "generator": "search",
                    "gsrsearch": f"{q} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                }, max_pages=2)
                for row in rows:
                    if row["norm_title"] not in seen_batch_titles and row["norm_title"] not in seen_titles:
                        seen_batch_titles.add(row["norm_title"])
                        candidates.append(row)
                time.sleep(sleep_sec)

        idx_counter = max(
            [int(Path(r["local_rel"]).stem.split("_")[-1]) for r in recs] + [0]
        ) + 1

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
                sha_raw = hashlib.sha256(resp.content).hexdigest()
                if sha_raw in seen_shas:
                    continue
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None or min(img.shape[:2]) < 150:
                    continue
                ah = compute_ahash(img)
                dh = compute_dhash(img)
                if any(
                    min(hex_hamming_distance(ah, o_ah), hex_hamming_distance(dh, o_dh)) <= 6
                    for _, o_ah, o_dh in seen_phashes
                ):
                    continue

                fname = f"cand_{idx_counter:03d}.jpg"
                out_p = group_dir / fname
                save_image_unicode(out_p, img)
                rel_p = out_p.relative_to(staging_root).as_posix()
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
                }
                recs.append(rec)
                seen_shas.add(rec["sha256"])
                seen_shas.add(sha_raw)
                seen_titles.add(cand["norm_title"])
                seen_phashes.append((rel_p, ah, dh))
                idx_counter += 1
                time.sleep(sleep_sec)
            except Exception:
                time.sleep(sleep_sec)

        existing_meta[group_key] = recs
        meta_file.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Collected v1c] {group_key}: {len(recs)}/{target}", flush=True)

        # Render contact sheet(s) in batches of 30 for clean inspection
        cs_dir = vl_p1c_dir / "manifests" / "contact_sheets_v1c" / "raw_pool"
        for page_i in range(0, len(recs), 30):
            chunk = recs[page_i:page_i + 30]
            cs_items = []
            for r in chunk:
                p = staging_root / r["local_rel"]
                try:
                    cs_items.append((f"{p.stem} | {r['title'][5:]}", load_image_unicode(p), None))
                except ValueError:
                    pass
            if cs_items:
                sfx = f"_p{page_i // 30 + 1}" if len(recs) > 30 else ""
                sheet_name = group_key.replace("/", "_") + sfx + ".jpg"
                make_contact_sheet(cs_items, cs_dir / sheet_name, cols=5, cell_w=260, cell_h=205)

    return existing_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    collect_staging_pool_v1c(
        root / "research" / "vl_p1",
        root / "research" / "vl_p1b",
        root / "research" / "vl_p1c",
        only_groups=args.only,
    )
