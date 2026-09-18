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
import hashlib, json, os, shutil, subprocess, sys, yaml

INPUT = Path('/kaggle/input')
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week8.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one Week 8 source bundle, found {bundles}'
    source_work = Path('/tmp/spectrashift-week8-source')
    if source_work.exists(): shutil.rmtree(source_work)
    shutil.unpack_archive(str(bundles[0]), str(source_work))
    projects = [source_work]
assert projects, 'No Week 8 source tree found'
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

def install_offline_foundation_dependencies():
    wheels = sorted(INPUT.rglob('foundation-wheels'))
    if not wheels: return
    missing = []
    for module, package in [('upath','universal-pathlib'), ('omegaconf','omegaconf'), ('iopath','iopath'), ('fvcore','fvcore'), ('einops','einops'), ('huggingface_hub','huggingface_hub')]:
        try: __import__(module)
        except ImportError: missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', str(wheels[0]), *missing])
"""


def _common_paths() -> str:
    return """MANIFEST = unique_file('partitions.parquet')
NORMALIZATION = unique_file('normalization.json')
FREEZE = unique_file('freeze_summary.json')
WEEK7_SUMMARY = unique_file('week7_run_summary.json')
LEDGER = unique_file('week8_checkpoint_ledger.csv')
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
"""


def prepare() -> dict[str, object]:
    setup = SETUP + """
WORK = Path('/kaggle/working/spectrashift-week8-evaluation-contracts')
WORK.mkdir(parents=True, exist_ok=True)
""" + _common_paths() + """WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
CANDIDATES = unique_file('evaluation_candidate_labels.parquet')
config = yaml.safe_load((PROJECT / 'configs/eval/week8.yaml').read_text())
config['paths'].update({
    'manifest_path': str(MANIFEST), 'staged_root': str(STAGED),
    'normalization_path': str(NORMALIZATION), 'freeze_summary_path': str(FREEZE),
    'week5_contracts_path': str(WEEK5_CONTRACTS), 'week7_summary_path': str(WEEK7_SUMMARY),
    'checkpoint_ledger_path': str(LEDGER), 'candidate_labels_path': str(CANDIDATES),
    'output_dir': str(WORK), 'evaluation_labels_path': str(WORK / 'evaluation_labels.parquet'),
    'evaluation_contract_path': str(WORK / 'evaluation_contract.json'),
    'support_contract_path': str(WORK / 'support_contract.json'),
})
RUNTIME_CONFIG = WORK / 'week8.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({'project': str(PROJECT), 'work': str(WORK), 'candidate_labels': str(CANDIDATES)})
"""
    run = """from spectrashift.train.week8 import freeze_week8_evaluation
summary = freeze_week8_evaluation(RUNTIME_CONFIG)
print(json.dumps(summary, indent=2))
assert summary['week8_sealing_complete']
assert summary['partition_counts'] == {'I': 3000, 'T-FI': 4000, 'T-PT': 4000}
assert summary['checkpoint_ledger_count'] == 111
assert summary['model_selection_frozen_before_label_access']
assert summary['evaluation_labels_loaded']
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 8: freeze sealed evaluation\nUse CPU with Internet off. Attach source v7, Week 2 frozen, Week 5 contracts, Week 7 complete, and the private evaluation-candidate dataset.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def seed_notebook(seed: int) -> dict[str, object]:
    output = f"spectrashift-week8-seed{seed}-eval"
    setup = SETUP + f"""
install_offline_foundation_dependencies()
SEED = {seed}
WORK = Path('/kaggle/working/{output}')
WORK.mkdir(parents=True, exist_ok=True)
""" + _common_paths() + """RGB_CONTRACT = unique_file('rgb_percentile_contract.json')
WEEK7_CONTRACTS = unique_file('week7_contracts_summary.json')
OLMO_CONTRACT = unique_file('olmoearth_input_contract.json')
EVAL_LABELS = unique_file('evaluation_labels.parquet')
EVAL_CONTRACT = unique_file('evaluation_contract.json')
SUPPORT_CONTRACT = unique_file('support_contract.json')
config = yaml.safe_load((PROJECT / 'configs/eval/week8.yaml').read_text())
config['paths'].update({
    'manifest_path': str(MANIFEST), 'staged_root': str(STAGED),
    'normalization_path': str(NORMALIZATION), 'freeze_summary_path': str(FREEZE),
    'week7_summary_path': str(WEEK7_SUMMARY), 'checkpoint_ledger_path': str(LEDGER),
    'evaluation_labels_path': str(EVAL_LABELS), 'evaluation_contract_path': str(EVAL_CONTRACT),
    'support_contract_path': str(SUPPORT_CONTRACT), 'rgb_contract_path': str(RGB_CONTRACT),
    'week7_contracts_path': str(WEEK7_CONTRACTS), 'olmo_contract_path': str(OLMO_CONTRACT),
})
RUNTIME_CONFIG = WORK / 'week8.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({'seed': SEED, 'project': str(PROJECT), 'work': str(WORK)})
"""
    run = """import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Select GPU T4 x2'
from spectrashift.train.week8 import evaluate_week8_seed
summary = evaluate_week8_seed(RUNTIME_CONFIG, SEED, [INPUT], WORK)
print(json.dumps({key: value for key, value in summary.items() if key != 'runs'}, indent=2))
assert summary['week8_seed_evaluation_complete']
assert summary['run_count'] == 37 and summary['prediction_domain_count'] == 111
assert summary['model_selection_after_label_access'] is False
assert summary['parameter_updates_after_label_access'] is False
"""
    return _notebook([
        _cell("markdown", f"# SpectraShift Week 8: seed {seed} final evaluation\nUse T4 x2 with Internet off and GPU 0. Attach source v7, Week 2 frozen, Week 6 contracts, Week 7 contracts v2, Week 7 complete, Week 8 evaluation contracts, and matching Week 5/6/7 seed datasets.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def aggregate() -> dict[str, object]:
    setup = SETUP + """
WORK = Path('/kaggle/working/spectrashift-week8-complete')
WORK.mkdir(parents=True, exist_ok=True)
""" + _common_paths() + """EVAL_LABELS = unique_file('evaluation_labels.parquet')
EVAL_CONTRACT = unique_file('evaluation_contract.json')
SUPPORT_CONTRACT = unique_file('support_contract.json')
seed_summaries = []
for seed in (17, 29, 43):
    matches = sorted(INPUT.rglob(f'week8_seed{seed}_summary.json'))
    assert len(matches) == 1, f'Expected one seed {seed} summary, found {matches}'
    seed_summaries.append(matches[0])
config = yaml.safe_load((PROJECT / 'configs/eval/week8.yaml').read_text())
config['paths'].update({
    'manifest_path': str(MANIFEST), 'staged_root': str(STAGED),
    'normalization_path': str(NORMALIZATION), 'freeze_summary_path': str(FREEZE),
    'week7_summary_path': str(WEEK7_SUMMARY), 'checkpoint_ledger_path': str(LEDGER),
    'evaluation_labels_path': str(EVAL_LABELS), 'evaluation_contract_path': str(EVAL_CONTRACT),
    'support_contract_path': str(SUPPORT_CONTRACT),
})
RUNTIME_CONFIG = WORK / 'week8.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
"""
    run = """from spectrashift.train.week8 import aggregate_week8
summary = aggregate_week8(RUNTIME_CONFIG, seed_summaries, WORK)
print(json.dumps(summary, indent=2))
assert summary['week8_complete'] and summary['week9_approved']
assert summary['evaluation_run_count'] == 111
assert summary['prediction_domain_count'] == 333
assert summary['checkpoint_ledger_count'] == 111
assert summary['evaluation_labels_loaded']
assert summary['model_selection_frozen_before_label_access']
assert summary['week9_implemented'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 8: aggregate final evaluation\nUse CPU with Internet off. Attach source v7, Week 7 complete, Week 8 evaluation contracts, and all three Week 8 seed outputs.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output-free Week 8 Kaggle notebooks")
    parser.add_argument("--output-dir", default="notebooks/kaggle")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "19_week8_prepare_evaluation.ipynb": prepare(),
        "20a_week8_evaluate_seed17.ipynb": seed_notebook(17),
        "20b_week8_evaluate_seed29.ipynb": seed_notebook(29),
        "20c_week8_evaluate_seed43.ipynb": seed_notebook(43),
        "21_week8_aggregate.ipynb": aggregate(),
    }
    for name, notebook in notebooks.items():
        path = root / name
        path.write_text(json.dumps(notebook, indent=1) + "\n")
        if path.stat().st_size >= 1_000_000:
            raise ValueError(f"Notebook exceeds Kaggle import limit: {path}")
        print({"path": str(path), "bytes": path.stat().st_size})


if __name__ == "__main__":
    main()
