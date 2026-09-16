from __future__ import annotations

import argparse
import json

from spectrashift.data.normalization import compute_normalization


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-band statistics using valid U pixels only")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(compute_normalization(args.config), indent=2))


if __name__ == "__main__":
    main()

