from __future__ import annotations

import argparse
import json

from spectrashift.train.week5 import run_week5_seed


def _mapping(values: list[str]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Week 5 downstream seed bundle")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contracts-summary", required=True)
    parser.add_argument("--pilot-summary", required=True)
    parser.add_argument("--encoder", action="append", default=[])
    parser.add_argument("--resume-root", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(run_week5_seed(
        args.config, args.seed, args.output, args.contracts_summary,
        args.pilot_summary, _mapping(args.encoder), args.resume_root,
    ), indent=2))


if __name__ == "__main__":
    main()
