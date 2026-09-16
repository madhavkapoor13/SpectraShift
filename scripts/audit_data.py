from __future__ import annotations

import argparse
import json

from spectrashift.data.audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit the SpectraShift Week 1 draft split")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.config), indent=2))


if __name__ == "__main__":
    main()

