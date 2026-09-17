from __future__ import annotations

import argparse
import json

from spectrashift.train.week6 import aggregate_week6


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate Week 6")
    parser.add_argument("--week5-summary", required=True)
    parser.add_argument("--week6-contracts", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_week6(
        args.week5_summary, args.week6_contracts, args.inputs, args.output
    ), indent=2))


if __name__ == "__main__":
    main()
