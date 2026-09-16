from __future__ import annotations

import argparse
import json

from spectrashift.train.ssl import train_ssl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SpectraShift VICReg encoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(train_ssl(args.config, args.seed, args.resume), indent=2))


if __name__ == "__main__":
    main()
