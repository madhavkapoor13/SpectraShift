from __future__ import annotations

import argparse
import json

from spectrashift.data.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Week 2 archive and staging capacity")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config), indent=2))


if __name__ == "__main__":
    main()
