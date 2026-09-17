from __future__ import annotations

import argparse
import json

from spectrashift.train.week5 import run_week5_pilots


def _mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, path = value.split("=", 1)
        result[key] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded Week 5 LR pilots")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contracts-summary", required=True)
    parser.add_argument("--encoder", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(run_week5_pilots(
        args.config, args.output, args.contracts_summary, _mapping(args.encoder)
    ), indent=2))


if __name__ == "__main__":
    main()
