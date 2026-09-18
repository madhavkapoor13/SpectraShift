from __future__ import annotations

import argparse
import json

from spectrashift.train.week9 import freeze_week9_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the immutable Week 9 analysis contract")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_week9_contracts(args.config), indent=2))


if __name__ == "__main__":
    main()
