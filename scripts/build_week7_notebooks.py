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
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week7.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one Week 7 source bundle, found {bundles}'
    source_work = Path('/tmp/spectrashift-week7-source')
    if source_work.exists(): shutil.rmtree(source_work)
    shutil.unpack_archive(str(bundles[0]), str(source_work))
    projects = [source_work]
assert projects, 'No Week 7 source tree found'
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
    wheels = sorted(INPUT.rglob('foundation-wheels')) + sorted(Path('/kaggle/working').rglob('foundation-wheels'))
    if not wheels: return
    missing = []
    for module, package in [('upath','universal-pathlib'), ('omegaconf','omegaconf'), ('iopath','iopath'), ('fvcore','fvcore'), ('einops','einops'), ('huggingface_hub','huggingface_hub')]:
        try: __import__(module)
        except ImportError: missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', str(wheels[0]), *missing])
"""


def prepare() -> dict[str, object]:
    setup = SETUP + """
from urllib.request import urlretrieve

WORK = Path('/kaggle/working/spectrashift-week7-contracts')
ASSETS = WORK / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
MANIFEST = unique_file('partitions.parquet')
NORMALIZATION = unique_file('normalization.json')
FREEZE = unique_file('freeze_summary.json')
WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
WEEK6_SUMMARY = unique_file('week6_run_summary.json')
RGB_CONTRACT = unique_file('rgb_percentile_contract.json')
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
config = yaml.safe_load((PROJECT / 'configs/downstream/week7.yaml').read_text())
config['data'].update({'manifest_path': str(MANIFEST), 'staged_root': str(STAGED), 'normalization_path': str(NORMALIZATION), 'freeze_summary_path': str(FREEZE)})
config['contracts'].update({'output_dir': str(WORK), 'week5_contracts_summary_path': str(WEEK5_CONTRACTS), 'week6_summary_path': str(WEEK6_SUMMARY), 'rgb_contract_path': str(RGB_CONTRACT)})

downloads = {
    'dinov2-source.zip': config['assets']['dinov2_source_url'],
    'dinov2_vits14_pretrain.pth': config['assets']['dinov2_weights_url'],
    'olmoearth-minimal.zip': config['assets']['olmoearth_minimal_url'],
}
for name, url in downloads.items():
    target = ASSETS / name
    if not target.exists(): urlretrieve(url, target)
for archive, directory in [('dinov2-source.zip','dinov2-source'), ('olmoearth-minimal.zip','olmoearth-minimal-source')]:
    target = ASSETS / directory
    if not target.exists():
        target.mkdir()
        shutil.unpack_archive(str(ASSETS / archive), str(target))
model_dir = ASSETS / 'olmoearth-v1_1-tiny'
model_dir.mkdir(exist_ok=True)
for name, key in [('config.json','olmoearth_config_url'), ('weights.pth','olmoearth_weights_url')]:
    target = model_dir / name
    if not target.exists(): urlretrieve(config['assets'][key], target)
wheels = ASSETS / 'foundation-wheels'
wheels.mkdir(exist_ok=True)
subprocess.check_call([sys.executable, '-m', 'pip', 'download', '-q', '--dest', str(wheels), 'universal-pathlib', 'omegaconf', 'fvcore', 'iopath', 'einops', 'huggingface_hub'])
RUNTIME_CONFIG = WORK / 'week7.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({'project': str(PROJECT), 'work': str(WORK), 'assets': str(ASSETS)})
"""
    run = """install_offline_foundation_dependencies()
from spectrashift.train.week7 import freeze_week7_contracts, validate_foundation_assets

summary = freeze_week7_contracts(RUNTIME_CONFIG, ASSETS)
print(json.dumps(summary, indent=2))
assert summary['week7_contracts_complete']
assert summary['dinov2']['display_name'] == 'DINOv2 ViT-S/14'
assert summary['dinov2']['register_tokens'] == 0
assert summary['olmoearth']['feature_dimension'] == 192
assert summary['evaluation_labels_loaded'] is False
smoke = validate_foundation_assets(RUNTIME_CONFIG, WEEK5_CONTRACTS, WORK / 'week7_contracts_summary.json')
print(json.dumps(smoke, indent=2))
assert smoke['week7_adapter_smoke_complete'] and smoke['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 7: freeze foundation contracts\nUse CPU with Internet enabled. Attach source v6, Week 2 frozen data, Week 5 contracts, Week 6 contracts, and Week 6 complete.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def _runtime_setup(output: str) -> str:
    return SETUP + f"""
install_offline_foundation_dependencies()
WORK = Path('/kaggle/working/{output}')
WORK.mkdir(parents=True, exist_ok=True)
MANIFEST = unique_file('partitions.parquet')
NORMALIZATION = unique_file('normalization.json')
FREEZE = unique_file('freeze_summary.json')
WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
WEEK7_CONTRACTS = unique_file('week7_contracts_summary.json')
RGB_CONTRACT = unique_file('rgb_percentile_contract.json')
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
config = yaml.safe_load((PROJECT / 'configs/downstream/week7.yaml').read_text())
config['data'].update({{'manifest_path': str(MANIFEST), 'staged_root': str(STAGED), 'normalization_path': str(NORMALIZATION), 'freeze_summary_path': str(FREEZE)}})
config['contracts'].update({{'week5_contracts_summary_path': str(WEEK5_CONTRACTS), 'rgb_contract_path': str(RGB_CONTRACT)}})
RUNTIME_CONFIG = WORK / 'week7.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({{'project': str(PROJECT), 'work': str(WORK)}})
"""


def pilots() -> dict[str, object]:
    run = """import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Select GPU T4 x2'
from spectrashift.train.week7 import run_week7_pilots

summary = run_week7_pilots(RUNTIME_CONFIG, WORK, WEEK5_CONTRACTS, WEEK7_CONTRACTS)
print(json.dumps(summary, indent=2))
assert summary['week7_pilots_complete'] and summary['foundation_matrix_approved']
assert summary['pilot_run_count'] == 6 and summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 7: foundation learning-rate pilots\nUse T4 x2 with Internet off and GPU 0. Attach source v6, Week 2 frozen data, Week 5 contracts, Week 6 contracts, and Week 7 contracts.\n"),
        _cell("code", _runtime_setup("spectrashift-week7-pilots")), _cell("code", run),
    ])


def seed_notebook(seed: int) -> dict[str, object]:
    output = f"spectrashift-week7-seed{seed}"
    setup = _runtime_setup(output) + f"""
SEED = {seed}
PILOTS = unique_file('week7_pilot_summary.json')
"""
    run = """import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Select GPU T4 x2'
from spectrashift.train.week7 import run_week7_seed

summary = run_week7_seed(RUNTIME_CONFIG, SEED, WORK, WEEK5_CONTRACTS, WEEK7_CONTRACTS, PILOTS, resume_roots=[INPUT])
print(json.dumps(summary, indent=2))
assert summary['week7_seed_complete'] and summary['run_count'] == 6
assert summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", f"# SpectraShift Week 7: seed {seed} foundation anchors\nUse T4 x2 with Internet off and GPU 0. Attach source v6, Week 2 frozen data, Week 5 contracts, Week 6 contracts, Week 7 contracts, Week 7 pilots, and any failed output from this seed.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def probes() -> dict[str, object]:
    setup = _runtime_setup("spectrashift-week7-probes") + "\nPILOTS = unique_file('week7_pilot_summary.json')\n"
    run = """import torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Select GPU T4 x2'
from spectrashift.train.week7 import run_week7_probes

summary = run_week7_probes(RUNTIME_CONFIG, WORK, WEEK5_CONTRACTS, WEEK7_CONTRACTS, PILOTS)
print(json.dumps({key: value for key, value in summary.items() if key not in {'linear_runs','knn_runs','feature_caches'}}, indent=2))
assert summary['foundation_feature_cache_count'] == 2
assert summary['foundation_linear_probe_count'] == 36
assert summary['foundation_knn_run_count'] == 6
assert summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 7: frozen foundation probes\nUse T4 x2 with Internet off and GPU 0. Attach source v6, Week 2 frozen data, Week 5 contracts, Week 6 contracts, Week 7 contracts, and Week 7 pilots.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def aggregate() -> dict[str, object]:
    setup = SETUP + """
WORK = Path('/kaggle/working/spectrashift-week7-complete')
WORK.mkdir(parents=True, exist_ok=True)
WEEK6_SUMMARY = unique_file('week6_run_summary.json')
CONTROLLED_CURVES = unique_file('controlled_curves.csv')
WEEK7_CONTRACTS = unique_file('week7_contracts_summary.json')
PILOTS = unique_file('week7_pilot_summary.json')
PROBES = unique_file('week7_probe_summary.json')
seed_paths = []
for seed in (17, 29, 43):
    matches = sorted(INPUT.rglob(f'week7_seed{seed}_summary.json'))
    assert len(matches) == 1, f'Expected one seed {seed} summary, found {matches}'
    seed_paths.append(matches[0])
"""
    run = """from spectrashift.train.week7 import aggregate_week7

summary = aggregate_week7(WEEK6_SUMMARY, CONTROLLED_CURVES, WEEK7_CONTRACTS, PILOTS, PROBES, seed_paths, WORK)
print(json.dumps({key: value for key, value in summary.items() if key not in {'foundation_runs','foundation_comparisons','week8_checkpoint_ledger'}}, indent=2))
assert summary['week7_complete'] and summary['week8_approved']
assert summary['foundation_full_run_count'] == 18
assert summary['foundation_linear_probe_count'] == 36
assert summary['foundation_knn_run_count'] == 6
assert summary['foundation_feature_cache_count'] == 2
assert summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 7: aggregate foundation baselines\nUse CPU with Internet off. Attach source v6, Week 6 complete, Week 7 contracts, pilots, probes, and all three Week 7 seed outputs.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output-free Week 7 Kaggle notebooks")
    parser.add_argument("--output-dir", default="notebooks/kaggle")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "14_week7_prepare_foundations.ipynb": prepare(),
        "15_week7_lr_pilots.ipynb": pilots(),
        "16a_week7_seed17.ipynb": seed_notebook(17),
        "16b_week7_seed29.ipynb": seed_notebook(29),
        "16c_week7_seed43.ipynb": seed_notebook(43),
        "17_week7_frozen_probes.ipynb": probes(),
        "18_week7_aggregate.ipynb": aggregate(),
    }
    for name, notebook in notebooks.items():
        path = root / name
        path.write_text(json.dumps(notebook, indent=1) + "\n")
        print({"path": str(path), "bytes": path.stat().st_size})


if __name__ == "__main__":
    main()
