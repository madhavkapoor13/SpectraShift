from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectrashift.train.week6 import run_week6_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one 16-job Week 6 seed bundle")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--week5-contracts", required=True)
    parser.add_argument("--pilots", required=True)
    parser.add_argument("--week5-summary", required=True)
    parser.add_argument("--week6-contracts", required=True)
    parser.add_argument("--encoder", action="append", default=[])
    parser.add_argument("--resume-root", action="append", default=[])
    args = parser.parse_args()
    encoders = {}
    for value in args.encoder:
        run_id, path = value.split("=", 1)
        encoders[run_id] = Path(path)
    result = run_week6_seed(
        args.config, args.seed, args.output, args.week5_contracts, args.pilots,
        args.week5_summary, args.week6_contracts, encoders, args.resume_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
