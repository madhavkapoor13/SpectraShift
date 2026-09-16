from __future__ import annotations

import argparse
import json

from spectrashift.train.probe import probe_ssl


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a frozen linear probe on a VICReg checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(probe_ssl(args.config, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
