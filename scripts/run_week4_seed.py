from __future__ import annotations

import argparse
import json

from spectrashift.train.week4 import run_week4_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one M2/M3/M4 Week 4 seed bundle")
    parser.add_argument("--config", action="append", required=True, help="Pass exactly three configs")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--week3-summary", required=True)
    parser.add_argument("--resume-root", action="append", default=[])
    args = parser.parse_args()
    result = run_week4_seed(
        args.config,
        args.output_root,
        args.week3_summary,
        args.resume_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
