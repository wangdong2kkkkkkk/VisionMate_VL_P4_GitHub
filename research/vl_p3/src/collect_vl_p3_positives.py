"""Collect fresh open-licensed white mobility cane (object_01) candidates for VL-P3.

Uses Wikimedia Commons, multilingual Wikipedia pages, and Openverse across open
research-compatible Creative Commons licenses (CC0, PDM, BY, BY-SA, BY-NC, BY-NC-SA),
with strict noise filtering (excluding moths, conferences, Brisbane bursts, fries,
non-cane guide dogs) and zero-leakage deduplication against all VL-P1..P2 holdouts & V3 negs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import cv2
import requests

from .collect_vl_p3_negatives import (
    COMMONS_API,
    OPENVERSE_API,
    USER_AGENT,
    _is_duplicate,
    _is_open_license,
    _strip_html,
    load_all_prior_signatures,
)
from .common_utils import (
    VL_P3_DIR,
    compute_ahash,
    compute_dhash,
    compute_sha256,
    load_image_unicode,
    make_contact_sheet,
    normalize_title,
    save_image_unicode,
)

EXCLUDE_POS_META_WORDS = (
    "logo", "icon", "svg", "diagram", "chart", "map", "flag", "coat_of_arms", "emblem",
    "seal", "stamp", "coin", "medal", "poster", "flyer", "cartoon", "comic", "drawing",
    "sketch", "painting", "engraving", "caricature", "vitrail", "stained", "sculpture",
    "statue", "tomb", "grave", "cemetery", "monument", "plaque", "road_sign", "signpost",
    "candy", "sugar", "sugarcane", "cane_sugar", "moth", "mythimna", "lepidoptera",
    "wainscot", "noctuidae", "insect", "butterfly", "caterpillar", "larva", "pupa",
    "entomology", "arachnid", "beetle", "mushroom", "plant", "flower", "botany",
    "tactile_paving", "paving_blocks", "braille_block", "fries", "goalball",
    "bicycle", "tandem", "ski", "trumpet", "saxophone", "orchestra", "choir", "concert",
    "conference", "seminar", "symposium", "roundtable", "press_conference", "podium",
    "microphone", "headshot", "official_portrait", "mugshot", "election", "ballot",
    "guitar", "piano", "violin", "cello", "drum", "mehrnews", "senator",
    "congressman", "braille", "magnifier", "ctet", "pool table", "billiard", "smoking",
    "cigarette", "bottle", "whisky", "church", "altar", "mosaic", "silhouette", "folio",
    "construction", "traffic cone", "safety cone", "pylon", "cane syrup", "steens",
    "raising cane", "fried chicken", "ice hockey", "manhole", "curtain", "blinds",
    "bamboo", "flute", "chopstick", "matson", "american colony", "library of congress",
    "jerusalem", "palestine", "optacon", "brailler", "embossed", "bookplate",
    "retinal", "glaucoma", "macular", "arteritis", "cataract", "spreadsheet",
    "excel", "screenshot", "clock", "hearing", "radio", "keyboard",
)

COMMONS_CATEGORIES = [
    "Category:White_canes",
    "Category:People_with_mobility_canes",
    "Category:White_Cane_Safety_Day",
    "Category:Blindenstock",
    "Category:Canne_blanche",
]

COMMONS_SEARCH_QUERIES = [
    '"white cane" -moth -mythimna -lepidoptera -mehrnews',
    '"white cane" street OR sidewalk OR crosswalk OR station OR walking -moth -mehrnews',
    'whitecane -moth -mehrnews',
    '"long cane" blind -moth',
    'blindenstock -moth -schild',
    'canne blanche aveugle -moth',
    '"bastone bianco"',
    '"baston blanco"',
    '"biała laska"',
    '"белая трость"',
    '"bengala branca"',
    '"白杖"',
    '"taststok"',
    '"slepecká hůl"',
]

WIKI_ARTICLES = [
    ("en", "White cane"),
    ("de", "Blindenstock"),
    ("fr", "Canne blanche"),
    ("es", "Bastón blanco"),
    ("it", "Bastone bianco"),
    ("pt", "Bengala branca"),
    ("ru", "Белая трость"),
    ("ja", "白杖"),
    ("nl", "Taststok"),
    ("pl", "Biała laska"),
    ("en", "Orientation and mobility"),
]

OPENVERSE_POS_QUERIES = [
    ("white cane", 15),
    ("blind cane", 12),
    ("mobility cane", 12),
    ("whitecane", 5),
    ("White Cane Day", 12),
    ("White Cane Safety Day", 10),
    ("walking with white cane", 10),
    ("white cane awareness", 10),
    ("person white cane", 5),
    ("tactile white cane", 6),
    ("visually impaired white cane", 6),
    ("blind walking street cane", 6),
    ("blindenstock", 6),
    ("canne blanche", 6),
    ("baston blanco", 5),
    ("bastone bianco", 5),
    ("long cane blind", 6),
    ("folding white cane", 5),
    ("red tip white cane", 5),
    ("white cane march", 4),
    ("walking cane blind", 4),
]

MAX_TARGET = 260


def _is_open_research_license(lic: str) -> bool:
    ll = (lic or "").strip().lower()
    if not ll:
        return False
    if _is_open_license(ll):
        return True
    first = re.split(r"[\s/,]+", ll)[0]
    return first in {"by", "by-sa", "by-nc", "by-nc-sa", "by-nd", "by-nc-nd", "cc0", "pdm", "pd", "gfdl", "fal", "ogl"}


def _is_clean_pos_candidate(title: str, desc: str = "", cats: str = "") -> bool:
    tl = (title or "").lower()
    if not tl.endswith((".jpg", ".jpeg", ".png", ".webp")):
        if not tl.startswith("openverse:"):
            return False
    blob = f"{title} {desc} {cats}".lower().replace("_", " ")
    if any(b.replace("_", " ") in blob for b in EXCLUDE_POS_META_WORDS):
        return False
    if "guide dog" in blob and not any(k in blob for k in ("cane", "canne", "stock", "baston", "bengala", "bastone", "lask", "трост", "杖", "stok")):
        return False
    if "tactile paving" in blob or "paving blocks" in blob or "braille block" in blob:
        if not any(k in blob for k in ("cane", "canne", "stock", "baston", "bengala", "bastone", "lask", "трост", "杖", "person", "walking")):
            return False
    return True


def _make_event_cluster(title: str, author: str) -> str:
    nt = normalize_title(title)
    nt_noburst = re.sub(r"\b\d{1,6}\b", "", nt).strip()
    nt_noburst = " ".join(nt_noburst.split()[:5])
    auth_tok = normalize_title(author)[:22] or "anon"
    return f"{auth_tok}|{nt_noburst}"


def _download_pos_image(session: requests.Session, item: dict, out_path: Path) -> dict | None:
    urls = [u for u in (item.get("image_url"), item.get("fallback_url")) if u]
    if not urls:
        return None
    for url in urls:
        for attempt in range(2):
            try:
                r = session.get(url, timeout=18)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code != 200 or len(r.content) < 5000:
                    break
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(".tmp")
                tmp_path.write_bytes(r.content)
                img = load_image_unicode(tmp_path)
                h, w = img.shape[:2]
                if min(h, w) < 200 or max(h, w) / max(1, min(h, w)) > 3.2:
                    tmp_path.unlink(missing_ok=True)
                    return None
                if max(h, w) > 960:
                    scale = 960.0 / max(h, w)
                    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                save_image_unicode(out_path, img)
                tmp_path.unlink(missing_ok=True)
                sha = compute_sha256(out_path)
                ah = compute_ahash(img)
                dh = compute_dhash(img)
                text_blob = f"{item.get('title', '')} {item.get('description', '')}".lower()
                has_person = any(t in text_blob for t in ("man", "woman", "person", "pedestrian", "walking", "holding", "student", "homme", "femme", "mann", "frau", "persona", "человек"))
                return {
                    "abs_path": str(out_path),
                    "sha256": sha,
                    "ahash": ah,
                    "dhash": dh,
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "title": item["title"],
                    "norm_title": item["norm_title"],
                    "source_cluster": item["source_cluster"],
                    "source_url": item["page_url"],
                    "image_url": url,
                    "license": item["license"],
                    "author": item["author"],
                    "description": item.get("description", ""),
                    "has_person": has_person,
                }
            except Exception:
                out_path.with_suffix(".tmp").unlink(missing_ok=True)
                time.sleep(0.3)
    return None


def _commons_pos_fetch(session: requests.Session, params: dict, max_pages: int = 3) -> list[dict]:
    out = []
    cur = dict(params)
    for _ in range(max_pages):
        req = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 960,
            "format": "json",
        }
        req.update(cur)
        data = None
        for attempt in range(3):
            try:
                r = session.get(COMMONS_API, params=req, timeout=18)
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                time.sleep(0.6)
        if not data:
            break
        for _, page in sorted(data.get("query", {}).get("pages", {}).items(), key=lambda kv: int(kv[0])):
            title = page.get("title", "")
            iis = page.get("imageinfo", [])
            if not iis:
                continue
            ii = iis[0]
            if ii.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            w = int(ii.get("thumbwidth") or ii.get("width") or 0)
            h = int(ii.get("thumbheight") or ii.get("height") or 0)
            if min(w, h) < 200:
                continue
            meta = ii.get("extmetadata", {})
            lic = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not _is_open_research_license(lic):
                continue
            desc = _strip_html(meta.get("ImageDescription", {}).get("value", ""))[:240]
            cats = _strip_html(meta.get("Categories", {}).get("value", ""))[:220]
            if not _is_clean_pos_candidate(title, desc, cats):
                continue
            nt = normalize_title(title)
            author = _strip_html(meta.get("Artist", {}).get("value", "Unknown"))[:120]
            out.append({
                "title": title,
                "norm_title": nt,
                "source_cluster": _make_event_cluster(title, author),
                "page_url": ii.get("descriptionurl", ""),
                "image_url": ii.get("thumburl") or ii.get("url"),
                "fallback_url": ii.get("url"),
                "license": lic,
                "author": author,
                "description": desc,
            })
        cont = data.get("continue")
        if not cont:
            break
        cur.update(cont)
    return out


def _openverse_pos_fetch(session: requests.Session, query: str, page: int = 1, page_size: int = 20) -> list[dict]:
    out = []
    for attempt in range(3):
        try:
            r = session.get(
                OPENVERSE_API,
                params={
                    "q": query,
                    "page": page,
                    "page_size": min(20, page_size),
                    "license": "cc0,pdm,by,by-sa,by-nc,by-nc-sa",
                },
                timeout=18,
            )
            if r.status_code == 429:
                time.sleep(2.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return []
            data = r.json()
            for row in data.get("results", []):
                lic = f"{row.get('license', '')} {row.get('license_version', '')}".strip()
                if not _is_open_research_license(lic):
                    continue
                title = str(row.get("title") or row.get("id") or "")
                nt = normalize_title(title)
                tags = " ".join(str(t.get("name", "")) for t in row.get("tags", []) if isinstance(t, dict))
                if not _is_clean_pos_candidate(f"Openverse:{title}", tags):
                    continue
                author = str(row.get("creator") or "Unknown")[:100]
                out.append({
                    "title": f"Openverse:{title[:100]}",
                    "norm_title": nt,
                    "source_cluster": _make_event_cluster(title or str(row.get("id", "")), author),
                    "page_url": row.get("foreign_landing_url") or row.get("url") or "",
                    "image_url": row.get("url"),
                    "fallback_url": row.get("thumbnail"),
                    "license": f"CC-{lic.upper()}" if lic.lower().startswith("by") else lic.upper(),
                    "author": author,
                    "description": tags[:180],
                })
            return out
        except Exception:
            time.sleep(0.5)
    return []


def _walk_commons_category_tree(session: requests.Session, root_cat: str, max_depth: int = 2) -> list[str]:
    """Recursively enumerate File:... titles under white-cane Wikimedia categories."""
    seen_cats: set[str] = set()
    file_titles: list[str] = []

    def _walk(cat: str, depth: int) -> None:
        if cat in seen_cats or depth < 0:
            return
        cl_l = cat.lower()
        if any(bad in cl_l for bad in ("candy cane", "du cane", "sugar", "rosa ")):
            return
        seen_cats.add(cat)
        cont: dict = {}
        for _ in range(8):
            req = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmtype": "file|subcat",
                "cmlimit": 100,
                "format": "json",
            }
            req.update(cont)
            try:
                r = session.get(COMMONS_API, params=req, timeout=15)
                if r.status_code != 200:
                    break
                data = r.json()
            except Exception:
                break
            for m in data.get("query", {}).get("categorymembers", []):
                if m.get("ns") == 14 and depth > 0:
                    _walk(m["title"], depth - 1)
                elif m.get("ns") == 6:
                    file_titles.append(m["title"])
            cont_tok = data.get("continue")
            if not cont_tok:
                break
            cont = cont_tok

    _walk(root_cat, max_depth)
    return list(dict.fromkeys(file_titles))


def collect_vl_p3_positives() -> Path:
    staging_v3 = VL_P3_DIR / "cache" / "staging_v3"
    out_dir = staging_v3 / "pos_new_canes"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = staging_v3 / "v3_positives_metadata.json"
    rej_path = staging_v3 / "rejected_pos_hashes.json"
    rej_urls_path = staging_v3 / "rejected_pos_urls.json"
    ver_ids_path = staging_v3 / "verified_new_pos_ids.json"
    existing = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else []
    rejected_shas = set(json.loads(rej_path.read_text(encoding="utf-8"))) if rej_path.is_file() else set()
    rejected_urls = set(json.loads(rej_urls_path.read_text(encoding="utf-8"))) if rej_urls_path.is_file() else set()
    already_verified_ids = set(json.loads(ver_ids_path.read_text(encoding="utf-8"))) if ver_ids_path.is_file() else set()

    seen_shas, seen_titles, seen_clusters, seen_phashes = load_all_prior_signatures()
    seen_shas.update(rejected_shas)
    # Remove short generic captions from blocking Openverse/Flickr photos
    seen_titles = {t for t in seen_titles if t.startswith("file ") or len(t) > 28}

    neg_meta = staging_v3 / "v3_negatives_metadata.json"
    if neg_meta.is_file():
        for rows in json.loads(neg_meta.read_text(encoding="utf-8")).values():
            for r in rows:
                seen_shas.add(r["sha256"].lower())
                if r.get("ahash") and r.get("dhash"):
                    seen_phashes.append((r["id"], r["ahash"], r["dhash"]))

    cluster_counts: dict[str, int] = {}
    recs = [r for r in existing if Path(r.get("abs_path", "")).is_file()]
    seen_page_urls = set(rejected_urls)
    for r in recs:
        seen_shas.add(r["sha256"].lower())
        if r.get("image_url"):
            seen_page_urls.add(r["image_url"])
        if r.get("source_url"):
            seen_page_urls.add(r["source_url"])
        cl = normalize_title(r.get("source_cluster", ""))
        cluster_counts[cl] = cluster_counts.get(cl, 0) + 1
        seen_phashes.append((r["id"], r["ahash"], r["dhash"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    cands: list[dict] = []
    seen_cand_urls: set[str] = set()

    def _add_cands(rows: list[dict]) -> None:
        for row in rows:
            u = row.get("image_url")
            pu = row.get("page_url", "")
            if not u or u in seen_cand_urls or u in seen_page_urls or (pu and pu in seen_page_urls):
                continue
            if row["title"].lower().startswith("file:") and row["norm_title"] in seen_titles:
                continue
            seen_cand_urls.add(u)
            cands.append(row)

    # 1. Recursive White Cane Commons category tree (high precision!)
    tree_titles: list[str] = []
    for cat in COMMONS_CATEGORIES:
        tree_titles.extend(_walk_commons_category_tree(session, cat, max_depth=2))
    tree_titles = list(dict.fromkeys(tree_titles))
    print(f"Recursive White Cane category tree unique files: {len(tree_titles)}", flush=True)
    for idx in range(0, len(tree_titles), 25):
        _add_cands(_commons_pos_fetch(session, {"titles": "|".join(tree_titles[idx:idx + 25])}, max_pages=1))

    # 2. Multilingual Wikipedia White Cane articles
    wiki_titles: list[str] = []
    for lang, article in WIKI_ARTICLES:
        try:
            r = session.get(f"https://{lang}.wikipedia.org/w/api.php",
                            params={"action": "query", "titles": article, "prop": "images", "imlimit": 50, "format": "json"}, timeout=12)
            for pg in r.json().get("query", {}).get("pages", {}).values():
                for im in pg.get("images", []):
                    t = im.get("title", "")
                    if t.lower().endswith((".jpg", ".jpeg", ".png")):
                        wiki_titles.append(t)
        except Exception:
            pass
    for idx in range(0, len(wiki_titles), 25):
        _add_cands(_commons_pos_fetch(session, {"titles": "|".join(wiki_titles[idx:idx + 25])}, max_pages=1))

    # 3. White-cane specific Commons searches
    for q in COMMONS_SEARCH_QUERIES:
        if len(cands) >= MAX_TARGET * 8:
            break
        _add_cands(_commons_pos_fetch(session, {"generator": "search", "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 50}, max_pages=4))

    # 4. Openverse white-cane queries (all open research CC licenses)
    for oq, max_pg in OPENVERSE_POS_QUERIES:
        if len(cands) >= MAX_TARGET * 14:
            break
        for pg in range(1, max_pg + 1):
            batch = _openverse_pos_fetch(session, oq, page=pg, page_size=20)
            _add_cands(batch)
            time.sleep(0.16)
            if len(batch) == 0:
                break

    def _rank_cand(c: dict) -> int:
        txt = f"{c.get('title','')} {c.get('description','')}".lower()
        exact_wc = any(t in txt for t in ("white cane", "whitecane", "long cane", "mobility cane", "canne blanche", "blindenstock", "bastone bianco", "baston blanco", "белая трость", "白杖", "biała laska"))
        person = any(t in txt for t in ("walking", "street", "pedestrian", "crosswalk", "man", "woman", "blind", "holding", "person", "commuter", "sidewalk", "day"))
        is_commons_wc = "commons" in c.get("page_url", "")
        return (4 if exact_wc else 0) + (2 if person else 0) + (1 if is_commons_wc else 0)

    cands.sort(key=_rank_cand, reverse=True)
    print(f"Total unique positive candidate URLs discovered: {len(cands)}", flush=True)

    max_existing_num = max([int(r["id"].split("_")[-1]) for r in recs], default=0)
    next_idx = max_existing_num + 1
    for cand in cands:
        if len(recs) >= MAX_TARGET:
            break
        cl = normalize_title(cand["source_cluster"])
        if cluster_counts.get(cl, 0) >= 4:
            continue
        rid = f"v3_pos_cane_{next_idx:03d}"
        dl = _download_pos_image(session, cand, out_dir / f"{rid}.jpg")
        if dl is None:
            continue
        if _is_duplicate(dl["sha256"], "", "", dl["ahash"], dl["dhash"], seen_shas, set(), set(), seen_phashes):
            Path(dl["abs_path"]).unlink(missing_ok=True)
            continue
        dl["id"] = rid
        seen_shas.add(dl["sha256"])
        seen_phashes.append((rid, dl["ahash"], dl["dhash"]))
        cluster_counts[cl] = cluster_counts.get(cl, 0) + 1
        recs.append(dl)
        next_idx += 1
        if len(recs) % 25 == 0:
            print(f"Collected {len(recs)} / {MAX_TARGET} new positive cane candidates...", flush=True)
            meta_path.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")

    meta_path.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    # Render ONLY newly collected unverified candidates so manual visual screening covers every new photo
    to_screen = [r for r in recs if r["id"] not in already_verified_ids]
    thumbs = [(f"{i:03d}:{r['id']}", load_image_unicode(Path(r["abs_path"]))) for i, r in enumerate(to_screen)]
    cs_dir = VL_P3_DIR / "cache" / "contact_sheets"
    for old in cs_dir.glob("screen_new_pos_p*.jpg"):
        old.unlink(missing_ok=True)
    for i in range(0, len(thumbs), 40):
        make_contact_sheet(thumbs[i:i + 40], cs_dir / f"screen_new_pos_p{i // 40 + 1}.jpg", cols=5, cell_w=240, cell_h=180)
    print(f"Done collecting new positives: {len(recs)} total ({len(to_screen)} new to screen), {(len(thumbs) + 39) // 40} sheets saved.")
    return meta_path


if __name__ == "__main__":
    p = collect_vl_p3_positives()
    print("Saved VL-P3 positives staging metadata:", p)
