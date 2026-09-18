from __future__ import annotations

import argparse
import json

from spectrashift.train.week8 import evaluate_week8_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen Week 8 seed bundle")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_week8_seed(args.config, args.seed, args.inputs, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
