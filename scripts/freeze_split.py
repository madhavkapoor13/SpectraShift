from __future__ import annotations

import argparse
import json

from spectrashift.data.freeze import freeze_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the footprint-audited SpectraShift split")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_split(args.config), indent=2))


if __name__ == "__main__":
    main()

