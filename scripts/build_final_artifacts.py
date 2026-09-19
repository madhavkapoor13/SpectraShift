from __future__ import annotations

import argparse
import json

from spectrashift.release import build_final_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen SpectraShift public result package")
    parser.add_argument("--week8-dir", required=True)
    parser.add_argument("--week9-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(build_final_artifacts(
        args.week8_dir, args.week9_dir, args.output_dir, args.root
    ), indent=2))


if __name__ == "__main__":
    main()
