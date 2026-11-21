"""Collect real open-licensed white cane photographs from Openverse (Flickr CC / Wikimedia / WordPress)."""
from __future__ import annotations

import json
from pathlib import Path
import time
import requests

from .collect_vl_p4_data import (
    EXCLUDE_POS_WORDS,
    USER_AGENT,
    _download_and_validate,
    load_all_prior_signatures,
)
from .common_utils import VL_P4_DIR, load_image_unicode, make_percent_grid_sheet

OV_OPEN_LICENSES = {"cc0", "pdm", "by", "by-sa", "by-nc", "by-nc-sa"}

OV_QUERIES = [
    "white cane blind",
    "blind person white cane",
    "white cane walking street",
    "white cane crosswalk",
    "visually impaired white cane",
    "folding white cane",
    "long cane blind mobility",
    "white cane safety day",
    "blindenstock",
    "canne blanche",
    "baston blanco",
    "bastone bianco",
    "blind man cane street",
    "blind woman white cane",
    "guide cane blind",
]


def _is_ov_open(lic: str) -> bool:
    return (lic or "").strip().lower() in OV_OPEN_LICENSES


def main() -> None:
    cache = VL_P4_DIR / "cache"
    raw = json.loads((cache / "raw_candidates_v4.json").read_text(encoding="utf-8"))
    sigs = load_all_prior_signatures()
    for r in raw["pos_records"] + raw["neg_records"]:
        sigs["shas"].add(r["sha256"])
        sigs["phashes"].append((r["ahash"], r["dhash"]))

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    ov_pos = []
    pos_idx = 3000
    for q in OV_QUERIES:
        if len(ov_pos) >= 350:
            break
        for page in range(1, 6):
            if len(ov_pos) >= 350:
                break
            try:
                r = sess.get(
                    "https://api.openverse.org/v1/images/",
                    params={"q": q, "page_size": 20, "page": page},
                    timeout=15,
                )
                if r.status_code == 429:
                    time.sleep(2.0)
                    continue
                if r.status_code != 200:
                    break
                results = r.json().get("results", [])
                if not results:
                    break
                for item in results:
                    if len(ov_pos) >= 350:
                        break
                    lic = str(item.get("license", "")).strip().lower()
                    if not _is_ov_open(lic):
                        continue
                    url = item.get("url", "")
                    title = str(item.get("title") or f"ov_{item.get('id', pos_idx)}")
                    pos_idx += 1
                    rec = _download_and_validate(
                        url, title, f"cc-{lic}", item.get("foreign_landing_url", ""),
                        cache / "raw_pos_v4", f"v4_raw_pos_{pos_idx:04d}",
                        sigs, sess, EXCLUDE_POS_WORDS,
                        {"label": "object_01", "query": f"ov:{q}"},
                    )
                    if rec:
                        ov_pos.append(rec)
                time.sleep(0.35)
            except Exception:
                break
        print(f"  Openverse '{q}' -> total ov_pos={len(ov_pos)}", flush=True)

    raw["pos_records"].extend(ov_pos)
    raw["pos_count"] = len(raw["pos_records"])
    (cache / "raw_candidates_v4.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    for i in range(0, len(ov_pos), 12):
        batch = ov_pos[i : i + 12]
        thumbs = [(r["id"], load_image_unicode(Path(r["abs_path"]))) for r in batch]
        make_percent_grid_sheet(thumbs, cache / "screening" / f"pos_screen_ov_{i // 12 + 1:02d}.jpg", cols=3, cell_w=380, cell_h=300)
    print(f"Finished Openverse collection: +{len(ov_pos)} candidates (total raw pos={raw['pos_count']})", flush=True)


if __name__ == "__main__":
    main()
