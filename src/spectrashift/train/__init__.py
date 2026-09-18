"""Training and checkpoint orchestration."""

from .probe import probe_ssl
from .pilots import select_week4_pilot
from .ssl import train_ssl
from .week4 import aggregate_week4, run_week4_seed, validate_week3_approval
from .downstream import cache_prefinetune_features, train_downstream
from .week5 import aggregate_week5, prepare_week5_contracts, run_week5_pilots, run_week5_seed
from .week7 import (
    aggregate_week7,
    freeze_week7_contracts,
    run_week7_pilots,
    run_week7_probes,
    run_week7_seed,
)

__all__ = [
    "aggregate_week4",
    "aggregate_week5",
    "aggregate_week7",
    "cache_prefinetune_features",
    "prepare_week5_contracts",
    "freeze_week7_contracts",
    "probe_ssl",
    "run_week4_seed",
    "run_week5_pilots",
    "run_week5_seed",
    "run_week7_pilots",
    "run_week7_probes",
    "run_week7_seed",
    "select_week4_pilot",
    "train_ssl",
    "train_downstream",
    "validate_week3_approval",
]
