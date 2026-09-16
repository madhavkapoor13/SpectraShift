from __future__ import annotations

import argparse
import json

from spectrashift.data.seal import seal_evaluation_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize frozen evaluation labels offline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(seal_evaluation_labels(args.config), indent=2))


if __name__ == "__main__":
    main()
