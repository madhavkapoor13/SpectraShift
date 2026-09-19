from __future__ import annotations

import argparse
import json

from spectrashift.release import verify_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the public SpectraShift release")
    parser.add_argument("--root", default=".")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_release(args.root, require_clean=not args.allow_dirty), indent=2))


if __name__ == "__main__":
    main()
