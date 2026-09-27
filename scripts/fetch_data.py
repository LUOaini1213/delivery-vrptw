"""Download the Solomon 100-customer instances and the SINTEF best-known-solution
(BKS) table + detailed BKS routes, and write data/MANIFEST.json with SHA-256 hashes.

The data is already committed under data/; this script documents where it came
from and can re-verify it:

    python scripts/fetch_data.py --verify     # hash-check the committed files
    python scripts/fetch_data.py --download   # re-download from SINTEF (network)

Source: SINTEF TOP, "Solomon benchmark, 100 customers"
https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/
(page retrieved 2026-09-27; the table there is updated over time).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGE = "https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/"
ZIP = "https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "delivery-vrptw/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_bks(page: str) -> dict:
    text = re.sub(r"<script.*?</script>", " ", page, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text)
    rows = {}
    for m in re.finditer(r"\b(rc|r|c)(\d{3})b?\s+(\d+)\s+(\d+\.\d+)\*?\s+([A-Z][A-Za-z0-9:]*)", text):
        name = (m.group(1) + m.group(2)).upper()
        if name not in rows:
            rows[name] = {"vehicles": int(m.group(3)), "distance": float(m.group(4)), "ref": m.group(5)}
    return rows


def download() -> None:
    page_b = fetch(PAGE)
    page = page_b.decode("utf-8", "ignore")
    bks = parse_bks(page)
    if len(bks) != 56:
        sys.exit(f"expected 56 BKS rows, parsed {len(bks)}")
    zb = fetch(ZIP)
    (DATA / "solomon").mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        for n in z.namelist():
            if n.endswith(".txt") and "/roadtopology/" not in n:
                b = z.read(n)
                out = DATA / "solomon" / Path(n).name.upper().replace(".TXT", ".txt")
                out.write_bytes(b)
                files[f"solomon/{out.name}"] = sha256(b)
    links = re.findall(r'href="(/contentassets/[^"]+\.txt)"', page)
    (DATA / "bks" / "solutions").mkdir(parents=True, exist_ok=True)
    for p in links:
        b = fetch("https://www.sintef.no" + p)
        name = Path(p).name.split("_")[0].replace(".txt", "").upper()
        out = DATA / "bks" / "solutions" / f"{name}.txt"
        out.write_bytes(b)
        files[f"bks/solutions/{out.name}"] = sha256(b)
    write_bks(bks)
    manifest = {"source_page": PAGE, "zip_url": ZIP, "zip_sha256": sha256(zb),
                "retrieved": "2026-09-27", "files": dict(sorted(files.items()))}
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"ok: {len(files)} files, zip sha256 {manifest['zip_sha256']}")


def write_bks(bks: dict) -> None:
    doc = {
        "source": PAGE,
        "retrieved": "2026-09-27",
        "objective": "hierarchical: 1) number of vehicles, 2) total distance",
        "convention": "Euclidean distance and travel time in double precision; "
                      "total distance rounded to two decimals (as stated on the SINTEF page)",
        "reference_codes_note": "Reference codes (RT, HG, MBD, ...) are the SINTEF page's citation keys; "
                                "full citations are listed on that page.",
        "instances": dict(sorted(bks.items())),
    }
    (DATA / "bks" / "solomon100_bks.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def verify() -> None:
    man = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [k for k, h in man["files"].items() if sha256((DATA / k).read_bytes()) != h]
    if bad:
        sys.exit(f"hash mismatch: {bad}")
    print(f"ok: {len(man['files'])} files match MANIFEST.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.download:
        download()
    verify()
