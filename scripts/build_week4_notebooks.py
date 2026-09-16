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


def seed_notebook(seed: int) -> dict[str, object]:
    output_name = f"spectrashift-week4-seed{seed}"
    setup = f"""from pathlib import Path
import json, os, shutil, sys, yaml

SEED = {seed}
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/{output_name}')
WORK.mkdir(parents=True, exist_ok=True)
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week4.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one source bundle, found {{bundles}}'
    project = WORK / 'source'
    shutil.unpack_archive(str(bundles[0]), str(project))
    projects = [project]
staging_summaries = sorted(INPUT.rglob('staging_summary.json'))
partition_manifests = sorted(INPUT.rglob('partitions.parquet'))
normalizations = sorted(INPUT.rglob('normalization.json'))
freeze_summaries = sorted(INPUT.rglob('freeze_summary.json'))
week3_summaries = sorted(INPUT.rglob('week3_run_summary.json'))
assert projects, 'No Week 4 source tree found'
projects.sort(key=lambda path: (0 if 'spectrashift-source' in str(path) else 1, len(str(path))))
assert len(staging_summaries) == len(partition_manifests) == 1
assert len(normalizations) == len(freeze_summaries) == len(week3_summaries) == 1
PROJECT = projects[0]
STAGED = staging_summaries[0].parent
MANIFEST = partition_manifests[0]
NORMALIZATION = normalizations[0]
FREEZE_SUMMARY = freeze_summaries[0]
WEEK3_SUMMARY = week3_summaries[0]
sys.path.insert(0, str(PROJECT / 'src'))
os.chdir(PROJECT)
print({{'seed': SEED, 'project': str(PROJECT), 'work': str(WORK)}})
"""
    contracts = """from spectrashift.train.week4 import validate_week3_approval, validate_week4_config

staging = json.loads(staging_summaries[0].read_text())
normalization = json.loads(NORMALIZATION.read_text())
freeze = json.loads(FREEZE_SUMMARY.read_text())
week3 = validate_week3_approval(WEEK3_SUMMARY)
assert staging['complete_patches'] == 50200 and staging['incomplete_patches'] == 0
assert normalization['sha256'] == '3b1d191beccde5b85fdced38c81984377e6fb3171683a6df6a9254c38f2a9a05'
assert freeze['training_approved']
assert freeze['manifest_sha256'] == '0b36a0c6c55f34a8963719af725096dde5ab1dbab72968c13639e31d1099000a'
assert week3['week4_selection']['selected']['learning_rate'] == 0.0001
runtime_configs = []
for model_id in ('m2', 'm3', 'm4'):
    filename = f'week4_{model_id}_seed{SEED}.yaml'
    config = yaml.safe_load((PROJECT / 'configs/ssl' / filename).read_text())
    config['data']['manifest_path'] = str(MANIFEST)
    config['data']['staged_root'] = str(STAGED)
    config['data']['normalization_path'] = str(NORMALIZATION)
    config['data']['freeze_summary_path'] = str(FREEZE_SUMMARY)
    config['run']['output_dir'] = str(WORK / config['run']['id'])
    config['run']['ledger_path'] = str(WORK / 'runs.jsonl')
    validate_week4_config(config)
    runtime = WORK / filename
    runtime.write_text(yaml.safe_dump(config, sort_keys=False))
    runtime_configs.append(runtime)
print([str(path) for path in runtime_configs])
"""
    gpu = """import torch
assert torch.cuda.is_available(), 'Select GPU T4 x2 before running'
gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
device_arch = f'sm_{major}{minor}'
supported_arches = torch.cuda.get_arch_list()
print({'gpu': gpu_name, 'device_arch': device_arch, 'pytorch_arches': supported_arches})
assert 'T4' in gpu_name, f'Select GPU T4 x2, found {gpu_name}'
assert device_arch in supported_arches, f'{device_arch} is unsupported by this PyTorch build'
"""
    run = """from spectrashift.train.week4 import run_week4_seed

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
summary = run_week4_seed(runtime_configs, WORK, WEEK3_SUMMARY, resume_roots=[INPUT])
print(json.dumps(summary, indent=2))
"""
    finish = f"""assert summary['week4_seed_complete'] is True and summary['seed'] == {seed}
assert len(summary['runs']) == 3
assert all(run['stability_gate'] and run['completion_gate'] for run in summary['runs'])
assert all(run['steps'] == 18720 and run['amp_overflow_skips'] == 0 for run in summary['runs'])
print({{'week4_seed_complete': True, 'seed': {seed}, 'summary_path': str(WORK / 'week4_seed{seed}_summary.json')}})
"""
    return _notebook([
        _cell("markdown", f"# SpectraShift Week 4: seed {seed}\nTrain M2, M3, and M4 sequentially on U. Attach the newest source dataset, `spectrashift-week2-frozen`, and `spectrashift-week3-pilots`. Use GPU T4 x2 with Internet disabled.\n"),
        _cell("code", setup),
        _cell("code", contracts),
        _cell("code", gpu),
        _cell("code", run),
        _cell("code", finish),
    ])


def aggregate_notebook() -> dict[str, object]:
    setup = """from pathlib import Path
import json, os, shutil, sys

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/spectrashift-week4-complete')
WORK.mkdir(parents=True, exist_ok=True)
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week4.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one source bundle, found {bundles}'
    project = WORK / 'source'
    shutil.unpack_archive(str(bundles[0]), str(project))
    projects = [project]
assert projects, 'No Week 4 source tree found'
projects.sort(key=lambda path: (0 if 'spectrashift-source' in str(path) else 1, len(str(path))))
PROJECT = projects[0]
sys.path.insert(0, str(PROJECT / 'src'))
os.chdir(PROJECT)
seed_summaries = sorted(INPUT.rglob('week4_seed*_summary.json'))
assert len(seed_summaries) == 3, f'Expected three seed summaries, found {seed_summaries}'
print([str(path) for path in seed_summaries])
"""
    aggregate = """from spectrashift.train.week4 import aggregate_week4

result = aggregate_week4(seed_summaries, WORK / 'week4_run_summary.json')
print(json.dumps(result, indent=2))
assert result['week4_complete'] and result['week5_approved']
assert result['run_count'] == 9 and result['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 4: aggregate and approve Week 5\nAttach the newest source dataset and all three private Week 4 seed datasets. This verification is CPU-only and requires no Internet.\n"),
        _cell("code", setup),
        _cell("code", aggregate),
    ])


def write_notebooks(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "05a_week4_seed17.ipynb": seed_notebook(17),
        "05b_week4_seed29.ipynb": seed_notebook(29),
        "05c_week4_seed43.ipynb": seed_notebook(43),
        "06_week4_aggregate.ipynb": aggregate_notebook(),
    }
    paths = []
    for name, value in artifacts.items():
        path = output_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output-free Week 4 Kaggle notebooks")
    parser.add_argument("--output-dir", default="notebooks/kaggle")
    args = parser.parse_args()
    for path in write_notebooks(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
