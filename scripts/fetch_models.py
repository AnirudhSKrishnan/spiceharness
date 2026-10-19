#!/usr/bin/env python3
"""Fetch pinned, unmodified vendor models declared by the local catalog.

Usage: python scripts/fetch_models.py [MODEL ...] [--force]
The catalog pins both archive and extracted-file SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "spiceharness" / "data" / "models"


def checked(data: bytes, expected: str, label: str) -> bytes:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {label}: expected {expected}, received {actual}")
    return data


def safe_member(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"Unsafe archive entry: {name!r}")
    return path.as_posix()


def model_bytes(payload: bytes, entry: str | None) -> bytes:
    if entry is None:
        return payload
    selected = safe_member(entry)
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                safe_member(member)
            return archive.read(selected)
    if payload.startswith(b"7z\xbc\xaf\x27\x1c"):
        import py7zr

        with tempfile.TemporaryDirectory(prefix="spiceharness-model-") as temporary:
            with py7zr.SevenZipFile(io.BytesIO(payload), mode="r") as archive:
                for member in archive.getnames():
                    safe_member(member)
                archive.extract(path=temporary, targets=[selected])
            extracted = Path(temporary) / selected
            if extracted.is_symlink() or not extracted.resolve().is_relative_to(Path(temporary).resolve()):
                raise ValueError("Archive member is a link outside the extraction directory")
            return extracted.read_bytes()
    raise ValueError("Catalog requested an archive entry but source is neither ZIP nor 7z")


def fetch(name: str, record: dict, destination: Path, force: bool = False) -> str:
    target = destination / safe_member(record["path"])
    if target.is_file() and not force:
        checked(target.read_bytes(), record["sha256"], name)
        return "already verified"
    request = urllib.request.Request(record["source_url"], headers={"User-Agent": "SpiceHarness/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(100 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        # Some vendor CDNs block urllib's TLS fingerprint. curl uses the same
        # pinned URL and bytes still pass both catalog hashes below.
        if exc.code != 403 or not shutil.which("curl"):
            raise
        fetched = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                  "--max-time", "45", "--max-filesize", str(100 * 1024 * 1024),
                                  record["source_url"]], capture_output=True, check=False, timeout=50)
        if fetched.returncode:
            raise ValueError(f"Vendor download failed for {name}; retry or download the pinned catalog URL manually") from exc
        payload = fetched.stdout
    if len(payload) > 100 * 1024 * 1024:
        raise ValueError("Model archive exceeds 100 MiB limit")
    if record.get("archive_sha256"):
        checked(payload, record["archive_sha256"], f"{name} archive")
    data = checked(model_bytes(payload, record.get("archive_entry")), record["sha256"], name)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return "downloaded and verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Catalog identifiers; default: all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--catalog", type=Path, default=MODEL_DIR / "catalog.json")
    parser.add_argument("--destination", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    selected = args.models or list(catalog)
    unknown = set(selected) - catalog.keys()
    if unknown:
        parser.error(f"Unknown model(s): {', '.join(sorted(unknown))}")
    try:
        for name in selected:
            print(f"{name}: {fetch(name, catalog[name], args.destination, args.force)}")
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Model fetch failed: {exc}\n")


if __name__ == "__main__":
    main()
