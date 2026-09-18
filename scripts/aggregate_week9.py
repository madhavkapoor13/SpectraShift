from __future__ import annotations

import argparse
import json

from spectrashift.train.week9 import aggregate_week9


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate Week 9")
    parser.add_argument("--config", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_week9(
        args.config, args.probes, args.inputs, args.output
    ), indent=2))


if __name__ == "__main__":
    main()
