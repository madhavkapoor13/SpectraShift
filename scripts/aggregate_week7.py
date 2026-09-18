from __future__ import annotations

import argparse
import json

from spectrashift.train.week7 import aggregate_week7


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate Week 7")
    parser.add_argument("--week6-summary", required=True)
    parser.add_argument("--controlled-curves", required=True)
    parser.add_argument("--week7-contracts", required=True)
    parser.add_argument("--pilots", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_week7(
        args.week6_summary, args.controlled_curves, args.week7_contracts,
        args.pilots, args.probes, args.inputs, args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
