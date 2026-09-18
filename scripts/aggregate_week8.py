from __future__ import annotations

import argparse
import json

from spectrashift.train.week8 import aggregate_week8


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate frozen Week 8 evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_week8(args.config, args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
