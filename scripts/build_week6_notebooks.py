from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(kind: str, source: str) -> dict[str, object]:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


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


PROJECT_SETUP = """from pathlib import Path
import hashlib, json, os, shutil, sys, yaml

INPUT = Path('/kaggle/input')
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week6.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one Week 6 source bundle, found {bundles}'
    source_work = Path('/tmp/spectrashift-week6-source')
    if source_work.exists():
        shutil.rmtree(source_work)
    shutil.unpack_archive(str(bundles[0]), str(source_work))
    projects = [source_work]
assert projects, 'No Week 6 source tree found'
projects.sort(key=lambda path: (0 if 'spectrashift-source' in str(path) else 1, len(str(path))))
PROJECT = projects[0]
sys.path.insert(0, str(PROJECT / 'src'))
os.chdir(PROJECT)

def unique_file(name):
    candidates = sorted(INPUT.rglob(name))
    by_hash = {}
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, path)
    assert len(by_hash) == 1, f'Expected one unique {name}; found {candidates}'
    return next(iter(by_hash.values()))
"""


def prepare_notebook() -> dict[str, object]:
    setup = PROJECT_SETUP + """
WORK = Path('/kaggle/working/spectrashift-week6-contracts')
WORK.mkdir(parents=True, exist_ok=True)
MANIFEST = unique_file('partitions.parquet')
NORMALIZATION = unique_file('normalization.json')
FREEZE = unique_file('freeze_summary.json')
WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
WEEK5_SUMMARY = unique_file('week5_run_summary.json')
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
config = yaml.safe_load((PROJECT / 'configs/downstream/week6.yaml').read_text())
config['data']['manifest_path'] = str(MANIFEST)
config['data']['staged_root'] = str(STAGED)
config['data']['normalization_path'] = str(NORMALIZATION)
config['data']['freeze_summary_path'] = str(FREEZE)
config['contracts']['output_dir'] = str(WORK)
config['contracts']['week5_contracts_summary_path'] = str(WEEK5_CONTRACTS)
config['contracts']['week5_summary_path'] = str(WEEK5_SUMMARY)
RUNTIME_CONFIG = WORK / 'week6.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({'project': str(PROJECT), 'work': str(WORK)})
"""
    run = """from spectrashift.train.week6 import freeze_week6_contracts

summary = freeze_week6_contracts(RUNTIME_CONFIG)
print(json.dumps(summary, indent=2))
assert summary['week6_contracts_complete']
assert summary['rgb_band_order'] == ['B04', 'B03', 'B02']
assert summary['rgb_fit_partition'] == 'U' and summary['rgb_fit_patch_count'] == 20000
assert summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 6: freeze the ImageNet RGB contract\nUse CPU with Internet off. Attach `spectrashift-source-v5`, `spectrashift-week2-frozen`, `spectrashift-week5-contracts`, and `spectrashift-week5-complete`.\n"),
        _cell("code", setup),
        _cell("code", run),
    ])


def seed_notebook(seed: int) -> dict[str, object]:
    output = f"spectrashift-week6-seed{seed}"
    setup = PROJECT_SETUP + f"""
SEED = {seed}
WORK = Path('/kaggle/working/{output}')
WORK.mkdir(parents=True, exist_ok=True)
MANIFEST = unique_file('partitions.parquet')
NORMALIZATION = unique_file('normalization.json')
FREEZE = unique_file('freeze_summary.json')
WEEK5_CONTRACTS = unique_file('week5_contracts_summary.json')
PILOT_SUMMARY = unique_file('week5_pilot_summary.json')
WEEK5_SUMMARY = unique_file('week5_run_summary.json')
WEEK6_CONTRACTS = unique_file('week6_contracts_summary.json')
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
config = yaml.safe_load((PROJECT / 'configs/downstream/week6.yaml').read_text())
config['data']['manifest_path'] = str(MANIFEST)
config['data']['staged_root'] = str(STAGED)
config['data']['normalization_path'] = str(NORMALIZATION)
config['data']['freeze_summary_path'] = str(FREEZE)
config['contracts']['week5_contracts_summary_path'] = str(WEEK5_CONTRACTS)
config['contracts']['week5_summary_path'] = str(WEEK5_SUMMARY)
RUNTIME_CONFIG = WORK / 'week6.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
import torch
encoder_paths = {{}}
for path in INPUT.rglob('encoder-final.pt'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_id') in {{'M2','M3','M4'}}:
        encoder_paths[str(payload['run_id'])] = path
expected = {{f'week4-{{model}}-seed{seed}' for model in ('m2','m3','m4')}}
assert set(encoder_paths) == expected, f'Expected {{expected}}, found {{set(encoder_paths)}}'
print({{'project': str(PROJECT), 'work': str(WORK), 'encoders': sorted(encoder_paths)}})
"""
    gpu = """assert torch.cuda.is_available(), 'Select GPU T4 x2 before running'
gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
assert 'T4' in gpu_name and f'sm_{major}{minor}' in torch.cuda.get_arch_list()
print({'gpu': gpu_name, 'gpu_count': torch.cuda.device_count(), 'using_device': 0})
"""
    run = """from spectrashift.train.week6 import run_week6_seed

summary = run_week6_seed(
    RUNTIME_CONFIG, SEED, WORK, WEEK5_CONTRACTS, PILOT_SUMMARY,
    WEEK5_SUMMARY, WEEK6_CONTRACTS, encoder_paths, resume_roots=[INPUT],
)
print(json.dumps(summary, indent=2))
assert summary['week6_seed_complete'] and summary['run_count'] == 16
assert summary['evaluation_labels_loaded'] is False
"""
    letter = {17: "a", 29: "b", 43: "c"}[seed]
    return _notebook([
        _cell("markdown", f"# SpectraShift Week 6: seed {seed}\nRun 15 missing curve jobs and the M1RGB control sequentially. Attach source v5, Week 2 frozen data, Week 5 contracts, the verified Week 5 pilot summary, Week 5 complete, Week 6 contracts, and `spectrashift-week4-seed{seed}`. Use GPU T4 x2 with Internet off. This is notebook 12{letter}.\n"),
        _cell("code", setup), _cell("code", gpu), _cell("code", run),
    ])


def aggregate_notebook() -> dict[str, object]:
    setup = PROJECT_SETUP + """
WORK = Path('/kaggle/working/spectrashift-week6-complete')
WORK.mkdir(parents=True, exist_ok=True)
archive_root = Path('/tmp/spectrashift-week6-aggregate-inputs')
if archive_root.exists():
    shutil.rmtree(archive_root)
archive_root.mkdir(parents=True)
for index, archive in enumerate(sorted(INPUT.rglob('*.zip'))):
    if archive.name != 'spectrashift-kaggle-source.zip':
        shutil.unpack_archive(str(archive), str(archive_root / str(index)))
roots = [INPUT, archive_root]

def candidates(name):
    return sorted({path for root in roots for path in root.rglob(name)})

week5_candidates = candidates('week5_run_summary.json')
week6_contract_candidates = candidates('week6_contracts_summary.json')
seed_candidates = candidates('week6_seed*_summary.json')
assert len(week5_candidates) == 1, week5_candidates
assert len(week6_contract_candidates) == 1, week6_contract_candidates
seed_groups = {}
for path in seed_candidates:
    payload = json.loads(path.read_text())
    if payload.get('week6_seed_complete') and payload.get('evaluation_labels_loaded') is False:
        seed_groups.setdefault(int(payload['seed']), []).append(path)
assert set(seed_groups) == {17, 29, 43}, f'Expected complete seeds 17, 29, 43; found {sorted(seed_groups)}'
SEED_SUMMARIES = []
for seed in (17, 29, 43):
    variants = seed_groups[seed]
    hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in variants}
    assert len(hashes) == 1, f'Conflicting summaries for seed {seed}: {variants}'
    SEED_SUMMARIES.append(variants[0])
WEEK5_SUMMARY = week5_candidates[0]
WEEK6_CONTRACTS = week6_contract_candidates[0]
print({'seed_summaries': [str(path) for path in SEED_SUMMARIES]})
"""
    run = """from spectrashift.train.week6 import aggregate_week6

summary = aggregate_week6(WEEK5_SUMMARY, WEEK6_CONTRACTS, SEED_SUMMARIES, WORK)
print(json.dumps(summary, indent=2))
assert summary['week6_complete'] and summary['week7_approved']
assert summary['new_run_count'] == 48
assert summary['controlled_run_count'] == 90 and summary['rgb_control_run_count'] == 3
assert summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 6: aggregate curves and approve Week 7\nUse CPU with Internet off. Attach source v5, `spectrashift-week5-complete`, `spectrashift-week6-contracts`, and all three Week 6 seed datasets. Direct datasets or downloaded result ZIPs are accepted.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def write_notebooks(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "11_week6_prepare_rgb.ipynb": prepare_notebook(),
        "12a_week6_seed17.ipynb": seed_notebook(17),
        "12b_week6_seed29.ipynb": seed_notebook(29),
        "12c_week6_seed43.ipynb": seed_notebook(43),
        "13_week6_aggregate.ipynb": aggregate_notebook(),
    }
    paths = []
    for name, value in artifacts.items():
        path = output_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output-free Week 6 Kaggle notebooks")
    parser.add_argument("--output-dir", default="notebooks/kaggle")
    args = parser.parse_args()
    for path in write_notebooks(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
