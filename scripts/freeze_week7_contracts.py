from __future__ import annotations

import argparse
import json

from spectrashift.train.week7 import freeze_week7_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze DINOv2 and OlmoEarth Week 7 contracts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_week7_contracts(args.config, args.assets), indent=2))


if __name__ == "__main__":
    main()
