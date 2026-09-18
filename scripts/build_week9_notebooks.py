from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(kind: str, source: str) -> dict[str, object]:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def _notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """from pathlib import Path
import hashlib, json, os, shutil, sys, yaml

INPUT = Path('/kaggle/input')
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week9.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one Week 9 source bundle, found {bundles}'
    source_work = Path('/tmp/spectrashift-week9-source')
    if source_work.exists(): shutil.rmtree(source_work)
    shutil.unpack_archive(str(bundles[0]), str(source_work))
    projects = [source_work]
assert projects, 'No Week 9 source tree found'
PROJECT = sorted(projects, key=lambda path: len(str(path)))[0]
sys.path.insert(0, str(PROJECT / 'src'))
os.chdir(PROJECT)

def unique_file(name):
    candidates = sorted(INPUT.rglob(name))
    by_hash = {}
    for path in candidates:
        by_hash.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)
    assert len(by_hash) == 1, f'Expected one unique {name}; found {candidates}'
    return next(iter(by_hash.values()))
"""


def _week2_paths() -> str:
    return """MANIFEST = unique_file('partitions.parquet')
NORMALIZATION = unique_file('normalization.json')
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
"""


def prepare() -> dict[str, object]:
    setup = SETUP + """
WORK = Path('/kaggle/working/spectrashift-week9-contracts')
WORK.mkdir(parents=True, exist_ok=True)
""" + _week2_paths() + """WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
WEEK7_SUMMARY = unique_file('week7_run_summary.json')
LEDGER = unique_file('week8_checkpoint_ledger.csv')
WEEK8_SUMMARY = unique_file('week8_run_summary.json')
EVAL_LABELS = unique_file('evaluation_labels.parquet')
EVAL_CONTRACT = unique_file('evaluation_contract.json')
SUPPORT_CONTRACT = unique_file('support_contract.json')
config = yaml.safe_load((PROJECT / 'configs/analysis/week9.yaml').read_text())
config['paths'].update({
    'manifest_path': str(MANIFEST), 'staged_root': str(STAGED),
    'normalization_path': str(NORMALIZATION), 'week5_contracts_path': str(WEEK5_CONTRACTS),
    'week7_summary_path': str(WEEK7_SUMMARY), 'checkpoint_ledger_path': str(LEDGER),
    'week8_summary_path': str(WEEK8_SUMMARY), 'week8_output_dir': str(WEEK8_SUMMARY.parent),
    'evaluation_labels_path': str(EVAL_LABELS), 'evaluation_contract_path': str(EVAL_CONTRACT),
    'support_contract_path': str(SUPPORT_CONTRACT), 'contracts_output_dir': str(WORK),
})
RUNTIME_CONFIG = WORK / 'week9.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({'project': str(PROJECT), 'work': str(WORK)})
"""
    run = """from spectrashift.train.week9 import freeze_week9_contracts
summary = freeze_week9_contracts(RUNTIME_CONFIG)
print(json.dumps(summary, indent=2))
assert summary['week9_contracts_complete']
assert summary['cka_patch_count'] == 1000
assert summary['nearest_neighbor_query_count'] == 100
assert summary['diagnostic_checkpoint_count'] == 6
assert summary['model_selection_after_week8'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 9: freeze analysis contracts\nUse CPU with Internet off. Attach source v8, Week 2 frozen, Week 5 contracts, Week 7 complete, Week 8 evaluation contracts, and Week 8 complete.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def probes() -> dict[str, object]:
    setup = SETUP + """
WORK = Path('/kaggle/working/spectrashift-week9-probes')
WORK.mkdir(parents=True, exist_ok=True)
""" + _week2_paths() + """WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
SUBSETS = unique_file('downstream_subsets.parquet')
WEEK7_SUMMARY = unique_file('week7_run_summary.json')
WEEK7_PROBES = unique_file('week7_probe_summary.json')
WEEK9_CONTRACTS = unique_file('week9_contracts_summary.json')
WEEK9_CONTRACT = unique_file('week9_contract.json')
config = yaml.safe_load((PROJECT / 'configs/analysis/week9.yaml').read_text())
config['paths'].update({
    'manifest_path': str(MANIFEST), 'staged_root': str(STAGED),
    'normalization_path': str(NORMALIZATION), 'week5_contracts_path': str(WEEK5_CONTRACTS),
    'subset_manifest_path': str(SUBSETS), 'week7_summary_path': str(WEEK7_SUMMARY),
    'week7_probe_summary_path': str(WEEK7_PROBES),
    'week9_contracts_summary_path': str(WEEK9_CONTRACTS),
    'week9_contract_path': str(WEEK9_CONTRACT),
})
RUNTIME_CONFIG = WORK / 'week9.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
"""
    run = """import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Select GPU T4 x2'
from spectrashift.train.week9 import run_week9_probes
summary = run_week9_probes(RUNTIME_CONFIG, [INPUT], WORK)
print(json.dumps({key: value for key, value in summary.items() if key not in {'linear_runs','knn_runs','feature_caches'}}, indent=2))
assert summary['week9_probes_complete']
assert summary['linear_probe_count'] == 108 and summary['new_linear_probe_count'] == 72
assert summary['knn_probe_count'] == 18 and summary['new_knn_probe_count'] == 12
assert summary['encoder_updates_during_week9'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 9: complete frozen representation probes\nUse T4 x2 with Internet off and GPU 0. Attach source v8, Week 2, Week 5 contracts and all three Week 5 seed datasets, Week 7 complete, Week 7 probes, and Week 9 contracts.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def diagnostics(seed: int) -> dict[str, object]:
    output = f"spectrashift-week9-diagnostics-seed{seed}"
    setup = SETUP + f"""
SEED = {seed}
WORK = Path('/kaggle/working/{output}')
WORK.mkdir(parents=True, exist_ok=True)
""" + _week2_paths() + """WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
WEEK9_CONTRACTS = unique_file('week9_contracts_summary.json')
WEEK9_CONTRACT = unique_file('week9_contract.json')
DIAGNOSTIC_LEDGER = unique_file('week9_diagnostic_ledger.csv')
EVAL_LABELS = unique_file('evaluation_labels.parquet')
SUPPORT_CONTRACT = unique_file('support_contract.json')
config = yaml.safe_load((PROJECT / 'configs/analysis/week9.yaml').read_text())
config['paths'].update({
    'manifest_path': str(MANIFEST), 'staged_root': str(STAGED),
    'normalization_path': str(NORMALIZATION), 'week5_contracts_path': str(WEEK5_CONTRACTS),
    'week9_contracts_summary_path': str(WEEK9_CONTRACTS),
    'week9_contract_path': str(WEEK9_CONTRACT), 'diagnostic_ledger_path': str(DIAGNOSTIC_LEDGER),
    'evaluation_labels_path': str(EVAL_LABELS), 'support_contract_path': str(SUPPORT_CONTRACT),
})
RUNTIME_CONFIG = WORK / 'week9.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
"""
    run = """import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Select GPU T4 x2'
from spectrashift.train.week9 import run_week9_diagnostics
summary = run_week9_diagnostics(RUNTIME_CONFIG, SEED, [INPUT], WORK)
print(json.dumps({key: value for key, value in summary.items() if key != 'runs'}, indent=2))
assert summary['week9_diagnostics_complete']
assert summary['diagnostic_checkpoint_count'] == 2
assert summary['stress_prediction_domain_count'] == 36
assert summary['new_stress_prediction_domain_count'] == 30
assert summary['nearest_neighbor_row_count'] == 1000
assert summary['encoder_updates_during_week9'] is False
"""
    return _notebook([
        _cell("markdown", f"# SpectraShift Week 9: seed {seed} diagnostics\nUse T4 x2 with Internet off and GPU 0. Attach source v8, Week 2 frozen, Week 5 contracts, Week 9 contracts, the matching Week 5 seed dataset, and matching Week 8 evaluation dataset.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def aggregate() -> dict[str, object]:
    setup = SETUP + """
WORK = Path('/kaggle/working/spectrashift-week9-complete')
WORK.mkdir(parents=True, exist_ok=True)
WEEK9_CONTRACTS = unique_file('week9_contracts_summary.json')
WEEK9_CONTRACT = unique_file('week9_contract.json')
EVAL_LABELS = unique_file('evaluation_labels.parquet')
SUPPORT_CONTRACT = unique_file('support_contract.json')
PROBES = unique_file('week9_probe_summary.json')
diagnostic_summaries = []
for seed in (17, 29, 43):
    matches = sorted(INPUT.rglob(f'week9_diagnostics_seed{seed}_summary.json'))
    assert len(matches) == 1, f'Expected one seed {seed} diagnostic summary, found {matches}'
    diagnostic_summaries.append(matches[0])
config = yaml.safe_load((PROJECT / 'configs/analysis/week9.yaml').read_text())
config['paths'].update({
    'week9_contracts_summary_path': str(WEEK9_CONTRACTS),
    'week9_contract_path': str(WEEK9_CONTRACT), 'evaluation_labels_path': str(EVAL_LABELS),
    'support_contract_path': str(SUPPORT_CONTRACT),
})
RUNTIME_CONFIG = WORK / 'week9.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
"""
    run = """from spectrashift.train.week9 import aggregate_week9
summary = aggregate_week9(RUNTIME_CONFIG, PROBES, diagnostic_summaries, WORK)
print(json.dumps(summary, indent=2))
assert summary['week9_complete'] and summary['week10_approved']
assert summary['linear_probe_count'] == 108 and summary['knn_probe_count'] == 18
assert summary['stress_prediction_domain_count'] == 108
assert summary['nearest_neighbor_row_count'] == 3000
assert summary['model_selection_after_week8'] is False
assert summary['encoder_updates_during_week9'] is False
assert summary['week10_implemented'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 9: aggregate representation and robustness analysis\nUse CPU with Internet off. Attach source v8, Week 8 complete, Week 9 contracts, Week 9 probes, and all three Week 9 diagnostic datasets.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output-free Week 9 Kaggle notebooks")
    parser.add_argument("--output-dir", default="notebooks/kaggle")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "22_week9_prepare_analysis.ipynb": prepare(),
        "23_week9_complete_probes.ipynb": probes(),
        "24a_week9_diagnostics_seed17.ipynb": diagnostics(17),
        "24b_week9_diagnostics_seed29.ipynb": diagnostics(29),
        "24c_week9_diagnostics_seed43.ipynb": diagnostics(43),
        "25_week9_aggregate.ipynb": aggregate(),
    }
    for name, notebook in notebooks.items():
        path = root / name
        path.write_text(json.dumps(notebook, indent=1) + "\n")
        if path.stat().st_size >= 1_000_000:
            raise ValueError(f"Notebook exceeds Kaggle import limit: {path}")
        print({"path": str(path), "bytes": path.stat().st_size})


if __name__ == "__main__":
    main()
