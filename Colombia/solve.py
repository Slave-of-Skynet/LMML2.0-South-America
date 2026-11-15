#!/usr/bin/env python3
"""LMML 2.0 - Colombia - "Looks Good to Me! - Bogota Sound Check".

Single-command, auditable solver (Python 3 standard library only).

    python solve.py --zip colombia.zip [--source reviews_Musical_Instruments_5.json.gz]

What it does
  1. Reads reviews.json straight from the task ZIP (nothing is extracted) and
     checks the dataset invariants.
  2. Scans every review for Unicode tampering: invisible format characters
     (zero-width space etc.) and non-Latin homoglyphs inside Latin words.
  3. Aligns every record (trusted and suspect) with the public source corpus
     (SNAP Amazon "Musical Instruments" 5-core reviews, McAuley et al.) and
     reports any record whose text or star rating differs from the original.
     The trusted rows are the false-positive control: they must all match.
  4. Builds the altered-index set, derives the SHA-256 mask exactly as the
     archive_note says, XORs it with all 2,200 archive tags and accepts only an
     exact ASCII SIGMOID_<PHRASE>. Every attempted index set is logged.

If --source is missing the public file is downloaded from SNAP and its
SHA-256 is checked before use.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

SOURCE_URL = ("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
              "reviews_Musical_Instruments_5.json.gz")
SOURCE_SHA256 = "03f9cc27ae13a454cb4ea6cb620fb10546cc1194ef921441d1c5f34addff2748"
FLAG_RE = re.compile(rb"SIGMOID_[A-Z0-9_]+")

# The task data applies one harmless style transform to whole records
# (straight apostrophe -> U+2019, hyphen -> U+2013). Undo it for alignment.
STYLE = {"’": "'", "–": "-"}
# Latin look-alikes from other scripts (only those needed are listed; any other
# non-Latin letter is still reported, it just cannot be mapped back).
CONFUSABLE = {"а": "a", "е": "e", "о": "o", "с": "c",
              "р": "p", "х": "x", "у": "y", "і": "i",
              "А": "A", "Е": "E", "О": "O", "С": "C",
              "Р": "P", "Х": "X", "ο": "o", "α": "a"}


def log(*a):
    print(*a, flush=True)


def load_task(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            log(f"[zip] {path}  members={names}")
            raw = z.read("reviews.json")
    else:
        with open(path, "rb") as f:
            raw = f.read()
    log(f"[data] reviews.json  bytes={len(raw)}  sha256={hashlib.sha256(raw).hexdigest()}")
    return json.loads(raw.decode("utf-8"))


def load_source(path):
    if not os.path.exists(path):
        log(f"[source] downloading {SOURCE_URL} -> {path}")
        urllib.request.urlretrieve(SOURCE_URL, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    log(f"[source] {path}  sha256={digest}")
    if digest != SOURCE_SHA256:
        sys.exit("[source] SHA-256 mismatch - refusing to use this file")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_invariants(d):
    T, S = d["trusted_reference"], d["samples"]
    idx = [r["sample_index"] for r in S]
    checks = {
        "trusted rows == 1000": len(T) == 1000,
        "sample rows == 1200": len(S) == 1200,
        "sample_index is a permutation of 0..1199": sorted(idx) == list(range(1200)),
        "sample_index == row position": idx == list(range(1200)),
        "rating is int 1..5": all(type(r["rating"]) is int and 1 <= r["rating"] <= 5 for r in T + S),
        "archive_tag is 64 hex chars": all(re.fullmatch(r"[0-9a-f]{64}", r["archive_tag"]) for r in T + S),
        "archive_tag values unique": len({r["archive_tag"] for r in T + S}) == 2200,
    }
    for k, v in checks.items():
        log(f"  [{'ok' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        sys.exit("dataset invariant failed")


def unicode_scan(rec):
    """Return list of (kind, field, position, codepoint, context)."""
    out = []
    for field in ("summary", "review_text"):
        s = rec[field]
        for m in re.finditer(r"[^\x00-\x7f]", s):
            ch, p = m.group(), m.start()
            if ch in STYLE:
                continue
            ctx = s[max(0, p - 12):p + 12]
            cat = unicodedata.category(ch)
            if cat == "Cf" or cat.startswith("Z"):
                kind = "invisible_char"
            elif cat.startswith("L") and not unicodedata.name(ch, "").startswith("LATIN"):
                kind = "homoglyph"
            else:
                kind = "other_unicode"
            out.append((kind, field, p, f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}", ascii(ctx)))
    # a record must use one style consistently (all ASCII or all typographic)
    t = rec["summary"] + rec["review_text"]
    if ("'" in t and "’" in t) or ("-" in t and "–" in t):
        out.append(("mixed_style", "both", -1, "", ""))
    return out


def canon(s):
    s = "".join(STYLE.get(c, CONFUSABLE.get(c, c)) for c in s)
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


def decode(rows, indices):
    key = ",".join(map(str, sorted(indices)))
    mask = hashlib.sha256(key.encode("utf-8")).digest()
    hits = []
    for coll, row, tag in rows:
        pt = bytes(a ^ b for a, b in zip(bytes.fromhex(tag), mask)).rstrip(b"\x00")
        if FLAG_RE.fullmatch(pt):
            hits.append((coll, row, pt.decode("ascii")))
    return key, mask.hex(), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="task ZIP (or reviews.json)")
    ap.add_argument("--source", default="reviews_Musical_Instruments_5.json.gz",
                    help="SNAP Musical Instruments 5-core file (downloaded if absent)")
    a = ap.parse_args()

    d = load_task(a.zip)
    T, S = d["trusted_reference"], d["samples"]
    log("[note] archive_note = " + d["archive_note"])
    log("[phase A1] dataset invariants")
    check_invariants(d)

    log("[phase A2] Unicode scan (both collections)")
    uni = collections.defaultdict(list)
    for coll, rows in (("trusted", T), ("samples", S)):
        for i, r in enumerate(rows):
            for f in unicode_scan(r):
                log(f"  {coll}[{i}] {f[0]:<14} {f[1]}@{f[2]} {f[3]}  ctx={f[4]}")
                if coll == "samples":
                    uni[f[0]].append(r["sample_index"])
                else:
                    sys.exit("unexpected anomaly in trusted_reference")

    log("[phase A3] alignment with the public source corpus")
    src = load_source(a.source)
    by_text = collections.defaultdict(set)
    for s in src:
        by_text[(s["summary"], s["reviewText"])].add(int(s["overall"]))
    log(f"  source reviews={len(src)}")
    rating_changed, unmatched = [], []
    text_changed = []
    for coll, rows in (("trusted", T), ("samples", S)):
        n_ok = 0
        for i, r in enumerate(rows):
            key = (canon(r["summary"]), canon(r["review_text"]))
            stars = by_text.get(key)
            if stars is None:
                unmatched.append((coll, i))
                log(f"  {coll}[{i}] NOT FOUND in source after canonicalisation")
                continue
            styled = (canon(r["summary"]) != "".join(STYLE.get(c, c) for c in r["summary"]) or
                      canon(r["review_text"]) != "".join(STYLE.get(c, c) for c in r["review_text"]))
            if styled:
                text_changed.append((coll, i))
            if r["rating"] not in stars:
                log(f"  {coll}[{i}] rating {r['rating']} but source says {sorted(stars)}  "
                    f"summary={ascii(r['summary'][:60])}")
                rating_changed.append((coll, i, sorted(stars)[0], r["rating"]))
            else:
                n_ok += 1
        log(f"  {coll}: {n_ok}/{len(rows)} rows match source rating")
    if unmatched or any(c == "trusted" for c, *_ in rating_changed + text_changed):
        sys.exit("trusted control failed or unmatched rows - stop")
    text_idx = sorted(i for c, i in text_changed)
    uni_idx = sorted(set(i for v in uni.values() for i in v))
    log(f"  rows whose text differs from source only by the detected characters: {text_idx}")
    assert text_idx == uni_idx, "Unicode scan and source diff disagree"

    zw = sorted(set(uni.get("invisible_char", [])))
    hg = sorted(set(uni.get("homoglyph", [])))
    up = sorted(i for c, i, o, n in rating_changed if n > o)
    down = sorted(i for c, i, o, n in rating_changed if n < o)
    other = sorted(set(uni.get("other_unicode", []) + uni.get("mixed_style", [])))
    log("[candidates]")
    log(f"  zero-width space ({len(zw)}): {zw}")
    log(f"  Cyrillic homoglyph ({len(hg)}): {hg}")
    log(f"  rating 3->4 ({len(up)}): {up}")
    log(f"  rating 4->3 ({len(down)}): {down}")
    log(f"  other ({len(other)}): {other}")

    tag_rows = [("trusted", i, r["archive_tag"]) for i, r in enumerate(T)] + \
               [("samples", r["sample_index"], r["archive_tag"]) for r in S]
    log("[phase B] decode attempts")
    attempts = [("zero-width only", zw), ("homoglyph only", hg),
                ("unicode (zw+homoglyph)", zw + hg), ("rating only", up + down),
                ("unicode + rating 3->4", zw + hg + up), ("unicode + rating 4->3", zw + hg + down),
                ("ALL altered", zw + hg + up + down + other)]
    final = None
    for name, ix in attempts:
        key, mhex, hits = decode(tag_rows, ix)
        log(f"  {name:<24} n={len(ix):<3} sha256={mhex} hits={hits}")
        log(f"      key_text={key}")
        if name == "ALL altered":
            final = (key, mhex, hits)
    key, mhex, hits = final
    if len(hits) != 1:
        sys.exit("NOT VERIFIED: expected exactly one flag hit")
    coll, row, flag = hits[0]
    log("[result]")
    log(f"  altered_indices = {key}")
    log(f"  mask_sha256     = {mhex}")
    rec = T[row] if coll == "trusted" else S[row]
    log(f"  carrier         = {coll}[{row}] archive_tag={rec['archive_tag']}")
    log(f"  FLAG            = {flag}")


if __name__ == "__main__":
    main()
