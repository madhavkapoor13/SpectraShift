from __future__ import annotations

import argparse
import json

from spectrashift.data.smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 2 data and tiny-overfit smoke gates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.max_steps), indent=2))


if __name__ == "__main__":
    main()

