"""High-precision photo collector for white cane positives using intitle + filemime:image/jpeg + Commons subcategories."""
from __future__ import annotations

import json
from pathlib import Path
import requests

from .collect_vl_p4_data import (
    EXCLUDE_POS_WORDS,
    USER_AGENT,
    _commons_fetch_pages,
    _download_and_validate,
    _strip_html,
    load_all_prior_signatures,
)
from .common_utils import VL_P4_DIR, load_image_unicode, make_percent_grid_sheet

PHOTO_EXCLUDE = EXCLUDE_POS_WORDS + (
    "federal_register", "archive", "public_law", "congress", "book", "page_", "folio",
    "gallica", "bnf", "cover", "title_page", "manuscript", "newspaper", "gazette",
)

INTITLE_QUERIES = [
    'filemime:image/jpeg intitle:white intitle:cane',
    'filemime:image/jpeg intitle:whitecane',
    'filemime:image/jpeg intitle:blindenstock',
    'filemime:image/jpeg intitle:canne intitle:blanche',
    'filemime:image/jpeg intitle:bastone intitle:bianco',
    'filemime:image/jpeg intitle:baston intitle:blanco',
    'filemime:image/jpeg intitle:blind intitle:cane',
    'filemime:image/jpeg intitle:blind intitle:walking',
    'filemime:image/jpeg intitle:visually intitle:impaired',
    'filemime:image/jpeg intitle:白杖',
    'filemime:image/jpeg intitle:taststok',
    'filemime:image/jpeg intitle:laska intitle:niewidomy',
    'filemime:image/jpeg intitle:трость intitle:белая',
    'filemime:image/jpeg "white cane" street OR crosswalk OR person OR walking',
    'filemime:image/jpeg "blind person" cane OR walking',
    'filemime:image/jpeg "long cane" orientation OR mobility OR blind',
    'filemime:image/jpeg "white cane safety day"',
    'filemime:image/jpeg "blindenstock" OR "blindenlangstock"',
]


def main() -> None:
    cache = VL_P4_DIR / "cache"
    raw = json.loads((cache / "raw_candidates_v4.json").read_text(encoding="utf-8"))
    sigs = load_all_prior_signatures()
    for r in raw["pos_records"] + raw["neg_records"]:
        sigs["shas"].add(r["sha256"])
        sigs["phashes"].append((r["ahash"], r["dhash"]))

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    new_pos = []
    pos_idx = 2000

    for q in INTITLE_QUERIES:
        if len(new_pos) >= 350:
            break
        for off in (0, 50, 100, 150, 200):
            if len(new_pos) >= 350:
                break
            pages = _commons_fetch_pages({
                "generator": "search",
                "gsrsearch": q,
                "gsrnamespace": 6,
                "gsrlimit": 50,
                "gsroffset": off,
            }, sess)
            if not pages:
                break
            for pg in pages:
                if len(new_pos) >= 350:
                    break
                ii = (pg.get("imageinfo") or [{}])[0]
                ext = ii.get("extmetadata", {})
                lic = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                url = ii.get("thumburl") or ii.get("url", "")
                title = pg.get("title", "")
                pos_idx += 1
                rec = _download_and_validate(
                    url, title, lic, ii.get("descriptionurl", ""),
                    cache / "raw_pos_v4", f"v4_raw_pos_{pos_idx:04d}",
                    sigs, sess, PHOTO_EXCLUDE,
                    {"label": "object_01", "query": q[:40]},
                )
                if rec:
                    new_pos.append(rec)
        print(f"  Query '{q[:42]}' -> total new_pos={len(new_pos)}", flush=True)

    raw["pos_records"].extend(new_pos)
    raw["pos_count"] = len(raw["pos_records"])
    (cache / "raw_candidates_v4.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    for i in range(0, len(new_pos), 12):
        batch = new_pos[i : i + 12]
        thumbs = [(r["id"], load_image_unicode(Path(r["abs_path"]))) for r in batch]
        make_percent_grid_sheet(thumbs, cache / "screening" / f"pos_screen_photo_{i // 12 + 1:02d}.jpg", cols=3, cell_w=380, cell_h=300)
    print(f"Finished intitle photo collection: +{len(new_pos)} candidates (total raw pos={raw['pos_count']})", flush=True)


if __name__ == "__main__":
    main()
