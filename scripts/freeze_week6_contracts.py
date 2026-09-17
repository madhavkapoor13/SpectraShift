from __future__ import annotations

import argparse
import json

from spectrashift.train.week6 import freeze_week6_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Week 6 U-only RGB percentile contract")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_week6_contracts(args.config), indent=2))


if __name__ == "__main__":
    main()
