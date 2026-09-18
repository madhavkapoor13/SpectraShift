from __future__ import annotations

import argparse
import json

from spectrashift.train.week7 import run_week7_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one six-job Week 7 seed bundle")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--week5-contracts", required=True)
    parser.add_argument("--week7-contracts", required=True)
    parser.add_argument("--pilots", required=True)
    parser.add_argument("--resume-root", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(run_week7_seed(
        args.config, args.seed, args.output, args.week5_contracts,
        args.week7_contracts, args.pilots, args.resume_root,
    ), indent=2))


if __name__ == "__main__":
    main()
