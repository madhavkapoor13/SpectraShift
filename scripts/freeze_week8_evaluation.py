from __future__ import annotations

import argparse
import json

from spectrashift.train.week8 import freeze_week8_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the sealed Week 8 evaluation contract")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_week8_evaluation(args.config), indent=2))


if __name__ == "__main__":
    main()
