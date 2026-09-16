from __future__ import annotations

import argparse
import json

from spectrashift.train.throughput import benchmark_throughput


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a provisional two-view Week 2 workload")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--timed-steps", type=int)
    args = parser.parse_args()
    print(json.dumps(benchmark_throughput(
        args.config, args.batch_size, args.warmup_steps, args.timed_steps
    ), indent=2))


if __name__ == "__main__":
    main()
