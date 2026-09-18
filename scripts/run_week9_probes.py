from __future__ import annotations

import argparse
import json

from spectrashift.train.week9 import run_week9_probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the frozen M1-M6 Week 9 probes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_week9_probes(args.config, args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
