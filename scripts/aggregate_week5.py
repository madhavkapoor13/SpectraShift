from __future__ import annotations

import argparse
import json

from spectrashift.train.week5 import aggregate_week5


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate Week 5 outputs")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--pilot-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_week5(args.input, args.pilot_summary, args.output), indent=2))


if __name__ == "__main__":
    main()
