from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PATTERNS = (
    "pyproject.toml",
    "src/**/*.py",
    "scripts/*.py",
    "configs/**/*.yaml",
    "configs/**/*.md",
    "notebooks/kaggle/*.ipynb",
    "notebooks/kaggle/README.md",
)


def source_files(root: Path) -> list[Path]:
    files = set()
    for pattern in PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_bundle(root: str | Path, output: str | Path) -> tuple[int, str]:
    root = Path(root).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files(root):
            relative = path.relative_to(root).as_posix()
            info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    temporary.replace(output)
    payload = output.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact Kaggle source dataset bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="dist/spectrashift-kaggle-source.zip")
    args = parser.parse_args()
    size, digest = build_bundle(args.root, args.output)
    print({"path": str(Path(args.output)), "bytes": size, "sha256": digest})


if __name__ == "__main__":
    main()
