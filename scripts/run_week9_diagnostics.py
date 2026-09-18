from __future__ import annotations

import argparse
import json

from spectrashift.train.week9 import run_week9_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Week 9 stress and representation diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(17, 29, 43))
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_week9_diagnostics(
        args.config, args.seed, args.inputs, args.output
    ), indent=2))


if __name__ == "__main__":
    main()
