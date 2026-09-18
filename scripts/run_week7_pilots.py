from __future__ import annotations

import argparse
import json

from spectrashift.train.week7 import run_week7_pilots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run six Week 7 foundation LR pilots")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--week5-contracts", required=True)
    parser.add_argument("--week7-contracts", required=True)
    args = parser.parse_args()
    print(json.dumps(run_week7_pilots(
        args.config, args.output, args.week5_contracts, args.week7_contracts
    ), indent=2))


if __name__ == "__main__":
    main()
