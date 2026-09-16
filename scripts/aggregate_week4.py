from __future__ import annotations

import argparse
import json

from spectrashift.train.week4 import aggregate_week4


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate all Week 4 SSL runs")
    parser.add_argument("--seed-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate_week4(args.seed_summary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
