"""Ensure distributions exclude repository-only data, archives, and caches."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check_archive(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            names = archive.getnames()
    else:
        raise ValueError(f"Unsupported distribution: {path.name}")
    forbidden = {"datasets", "legacy_v4", "work", ".venv", ".weather-cache", "__pycache__", ".git"}
    bad = []
    for name in names:
        member = PurePosixPath(name)
        if (
            forbidden.intersection(member.parts)
            or member.suffix in {".pyc", ".xlam", ".xls", ".xlsm", ".xlsx"}
            or member.name.lower().startswith("school_")
        ):
            bad.append(name)
    if bad:
        raise ValueError(f"Repository-only or generated files in {path.name}: {bad}")
    if not names:
        raise ValueError(f"Empty distribution: {path.name}")
    print(f"{path.name}: {len(names)} entries; release boundary verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        for path in args.archives:
            check_archive(path)
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"release check: {exc}\n")


if __name__ == "__main__":
    main()
