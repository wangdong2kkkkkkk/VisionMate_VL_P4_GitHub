"""Supplement walking_stick, trekking_pole, and clean white-cane candidates for VL-P4."""
from __future__ import annotations

import json
from pathlib import Path
import requests

from .collect_vl_p4_data import (
    EXCLUDE_NEG_WORDS,
    EXCLUDE_POS_WORDS,
    USER_AGENT,
    _commons_fetch_pages,
    _download_and_validate,
    _strip_html,
    load_all_prior_signatures,
)
from .common_utils import VL_P4_DIR, load_image_unicode, make_percent_grid_sheet

EXTRA_POS_EXCLUDE = EXCLUDE_POS_WORDS + (
    "federal_register", "federal register", "archive", "public_law", "congress",
    "bill", "statute", "dixie", "mountebank", "hamilton", "journal",
)


def main() -> None:
    cache = VL_P4_DIR / "cache"
    raw = json.loads((cache / "raw_candidates_v4.json").read_text(encoding="utf-8"))
    sigs = load_all_prior_signatures()
    for r in raw["pos_records"] + raw["neg_records"]:
        sigs["shas"].add(r["sha256"])
        sigs["phashes"].append((r["ahash"], r["dhash"]))

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    extra_negs = []
    neg_idx = len(raw["neg_records"]) + 300
    for grp, queries in [
        ("walking_stick_confuser", ['"walking stick" person OR hiker', '"walking stick" wooden OR cane', 'crutches walking street']),
        ("trekking_pole_confuser", ['"trekking poles"', '"hiking poles"', '"nordic walking"']),
    ]:
        cnt = sum(1 for r in raw["neg_records"] if r["confuser_group"] == grp)
        for q in queries:
            for off in (0, 50, 100):
                if cnt >= 85:
                    break
                pages = _commons_fetch_pages({"generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": off}, sess)
                for pg in pages:
                    if cnt >= 85:
                        break
                    ii = (pg.get("imageinfo") or [{}])[0]
                    ext = ii.get("extmetadata", {})
                    lic = _strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                    url = ii.get("thumburl") or ii.get("url", "")
                    title = pg.get("title", "")
                    neg_idx += 1
                    rec = _download_and_validate(
                        url, title, lic, ii.get("descriptionurl", ""),
                        cache / "raw_neg_v4" / grp, f"v4_raw_neg_{neg_idx:04d}",
                        sigs, sess, EXCLUDE_NEG_WORDS,
                        {"label": "unknown", "confuser_group": grp, "person_holding_confuser": True},
                    )
                    if rec:
                        extra_negs.append(rec)
                        cnt += 1
        print(f"Supplemented {grp}: total={cnt}", flush=True)

    extra_pos = []
    pos_idx = len(raw["pos_records"]) + 300
    pos_queries = [
        'blindenstock -pdf -book -archive',
        '"canne blanche" -pdf -book',
        '"white cane" walking OR blind -pdf -federal -register -moth',
        '"白杖"',
        '"bastone bianco"',
        '"baston blanco"',
        '"biała laska"',
        '"белая трость"',
        '"taststok"',
    ]
    for q in pos_queries:
        for off in (0, 50, 100, 150, 200):
            if len(extra_pos) >= 220:
                break
            pages = _commons_fetch_pages({"generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 50, "gsroffset": off}, sess)
            if not pages:
                break
            for pg in pages:
                if len(extra_pos) >= 220:
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
                    sigs, sess, EXTRA_POS_EXCLUDE,
                    {"label": "object_01", "query": q[:36]},
                )
                if rec:
                    extra_pos.append(rec)
        print(f"After extra pos {q[:28]}: +{len(extra_pos)}", flush=True)

    raw["pos_records"].extend(extra_pos)
    raw["neg_records"].extend(extra_negs)
    raw["pos_count"] = len(raw["pos_records"])
    raw["neg_count"] = len(raw["neg_records"])
    (cache / "raw_candidates_v4.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    for i in range(0, len(extra_pos), 12):
        batch = extra_pos[i : i + 12]
        thumbs = [(r["id"], load_image_unicode(Path(r["abs_path"]))) for r in batch]
        make_percent_grid_sheet(thumbs, cache / "screening" / f"pos_screen_ex_{i // 12 + 1:02d}.jpg", cols=3, cell_w=380, cell_h=300)
    print(f"Done! Total raw pos={raw['pos_count']}, neg={raw['neg_count']}", flush=True)


if __name__ == "__main__":
    main()
