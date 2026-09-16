from __future__ import annotations

import argparse
import json

from spectrashift.data.archive import stage_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage selected BigEarthNet v2 candidates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(stage_archive(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()

