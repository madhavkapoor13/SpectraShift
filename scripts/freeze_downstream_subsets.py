from __future__ import annotations

import argparse
import json

from spectrashift.data.downstream import freeze_downstream_subsets
from spectrashift.train.week5 import prepare_week5_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Week 5 nested downstream subsets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-contracts", action="store_true")
    args = parser.parse_args()
    result = prepare_week5_contracts(args.config) if args.prepare_contracts else freeze_downstream_subsets(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
