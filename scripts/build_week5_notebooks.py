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
import json, os, shutil, sys, yaml

INPUT = Path('/kaggle/input')
projects = [p.parent for p in INPUT.rglob('pyproject.toml') if (p.parent / 'src/spectrashift/train/week5.py').is_file()]
if not projects:
    bundles = sorted(INPUT.rglob('spectrashift-kaggle-source.zip'))
    assert len(bundles) == 1, f'Expected one Week 5 source bundle, found {bundles}'
    source_work = Path('/tmp/spectrashift-week5-source')
    if source_work.exists():
        shutil.rmtree(source_work)
    shutil.unpack_archive(str(bundles[0]), str(source_work))
    projects = [source_work]
assert projects, 'No Week 5 source tree found'
projects.sort(key=lambda path: (0 if 'spectrashift-source' in str(path) else 1, len(str(path))))
PROJECT = projects[0]
sys.path.insert(0, str(PROJECT / 'src'))
os.chdir(PROJECT)
"""


def prepare_notebook() -> dict[str, object]:
    setup = PROJECT_SETUP + """
WORK = Path('/kaggle/working/spectrashift-week5-contracts')
WORK.mkdir(parents=True, exist_ok=True)
manifests = sorted(INPUT.rglob('partitions.parquet'))
normalizations = sorted(INPUT.rglob('normalization.json'))
freeze_summaries = sorted(INPUT.rglob('freeze_summary.json'))
week4_summaries = sorted(INPUT.rglob('week4_run_summary.json'))
assert len(manifests) == len(normalizations) == len(freeze_summaries) == len(week4_summaries) == 1
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
config = yaml.safe_load((PROJECT / 'configs/downstream/week5.yaml').read_text())
config['data']['manifest_path'] = str(manifests[0])
config['data']['staged_root'] = str(STAGED)
config['data']['normalization_path'] = str(normalizations[0])
config['data']['freeze_summary_path'] = str(freeze_summaries[0])
config['contracts']['output_dir'] = str(WORK)
config['contracts']['week4_summary_path'] = str(week4_summaries[0])
config['contracts']['subset_manifest_path'] = str(WORK / 'downstream_subsets.parquet')
config['contracts']['downstream_contract_path'] = str(WORK / 'downstream_contract.json')
config['contracts']['imagenet_weights_path'] = str(WORK / 'resnet18-f37072fd.pth')
RUNTIME_CONFIG = WORK / 'week5.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print({'project': str(PROJECT), 'work': str(WORK)})
"""
    run = """from spectrashift.train.week5 import prepare_week5_contracts

summary = prepare_week5_contracts(RUNTIME_CONFIG)
print(json.dumps(summary, indent=2))
assert summary['week5_contracts_complete']
assert len(summary['supported_class_indices']) == 16
assert summary['imagenet_weights_sha256'].startswith('f37072fd')
assert summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 5: freeze downstream contracts\nUse CPU with Internet enabled. Attach `spectrashift-source-v4`, `spectrashift-week2-frozen`, and `spectrashift-week4-complete`.\n"),
        _cell("code", setup),
        _cell("code", run),
    ])


def _runtime_setup(output_name: str, seed: int | None = None) -> str:
    seed_line = f"SEED = {seed}\n" if seed is not None else ""
    return PROJECT_SETUP + seed_line + f"""
WORK = Path('/kaggle/working/{output_name}')
WORK.mkdir(parents=True, exist_ok=True)
manifests = sorted(INPUT.rglob('partitions.parquet'))
normalizations = sorted(INPUT.rglob('normalization.json'))
freeze_summaries = sorted(INPUT.rglob('freeze_summary.json'))
contracts_summaries = sorted(INPUT.rglob('week5_contracts_summary.json'))
week4_summaries = sorted(INPUT.rglob('week4_run_summary.json'))
assert len(manifests) == len(normalizations) == len(freeze_summaries) == len(contracts_summaries) == len(week4_summaries) == 1
STAGED = next(path.parent for path in INPUT.rglob('staging_summary.json'))
CONTRACTS_SUMMARY = contracts_summaries[0]
CONTRACTS = json.loads(CONTRACTS_SUMMARY.read_text())
from spectrashift.train.common import file_sha256
assert file_sha256(week4_summaries[0]) == CONTRACTS['week4_summary_sha256']
config = yaml.safe_load((PROJECT / 'configs/downstream/week5.yaml').read_text())
config['data']['manifest_path'] = str(manifests[0])
config['data']['staged_root'] = str(STAGED)
config['data']['normalization_path'] = str(normalizations[0])
config['data']['freeze_summary_path'] = str(freeze_summaries[0])
RUNTIME_CONFIG = WORK / 'week5.yaml'
RUNTIME_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
import torch
encoder_paths = {{}}
for path in INPUT.rglob('encoder-final.pt'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_id') in {{'M2','M3','M4'}}:
        encoder_paths[str(payload['run_id'])] = path
print({{'project': str(PROJECT), 'work': str(WORK), 'encoders': sorted(encoder_paths)}})
"""


def pilot_notebook() -> dict[str, object]:
    setup = _runtime_setup("spectrashift-week5-pilots")
    gpu = """assert torch.cuda.is_available(), 'Select GPU T4 x2 before running'
gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
assert 'T4' in gpu_name and f'sm_{major}{minor}' in torch.cuda.get_arch_list()
assert set(encoder_paths) == {f'week4-{model}-seed17' for model in ('m2','m3','m4')}
print({'gpu': gpu_name, 'gpu_count': torch.cuda.device_count()})
"""
    run = """from spectrashift.train.week5 import run_week5_pilots

summary = run_week5_pilots(RUNTIME_CONFIG, WORK, CONTRACTS_SUMMARY, encoder_paths)
print(json.dumps(summary, indent=2))
assert summary['week5_pilots_complete'] and summary['week5_final_approved']
assert summary['run_count'] == 15 and summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 5: bounded LR pilots\nAttach the source, Week 2 frozen data, Week 4 aggregate, Week 5 contracts, and `spectrashift-week4-seed17`. Use GPU T4 x2 with Internet off.\n"),
        _cell("code", setup), _cell("code", gpu), _cell("code", run),
    ])


def seed_notebook(seed: int) -> dict[str, object]:
    output = f"spectrashift-week5-seed{seed}"
    setup = _runtime_setup(output, seed)
    gpu = f"""assert torch.cuda.is_available(), 'Select GPU T4 x2 before running'
gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
assert 'T4' in gpu_name and f'sm_{{major}}{{minor}}' in torch.cuda.get_arch_list()
expected = {{f'week4-{{model}}-seed{seed}' for model in ('m2','m3','m4')}}
assert set(encoder_paths) == expected, f'Expected {{expected}}, found {{set(encoder_paths)}}'
pilot_summaries = sorted(INPUT.rglob('week5_pilot_summary.json'))
assert len(pilot_summaries) == 1
PILOT_SUMMARY = pilot_summaries[0]
print({{'gpu': gpu_name, 'gpu_count': torch.cuda.device_count()}})
"""
    run = """from spectrashift.train.week5 import run_week5_seed

summary = run_week5_seed(
    RUNTIME_CONFIG, SEED, WORK, CONTRACTS_SUMMARY, PILOT_SUMMARY,
    encoder_paths, resume_roots=[INPUT],
)
print(json.dumps(summary, indent=2))
assert summary['week5_seed_complete'] and summary['run_count'] == 15
assert len(summary['feature_caches']) == 4 and summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", f"# SpectraShift Week 5: seed {seed}\nRun the 15 M0-M4 anchor jobs sequentially. Attach the source, Week 2 data, Week 4 aggregate, Week 5 contracts/pilots, and `spectrashift-week4-seed{seed}`. Use GPU T4 x2 with Internet off.\n"),
        _cell("code", setup), _cell("code", gpu), _cell("code", run),
    ])


def aggregate_notebook() -> dict[str, object]:
    setup = PROJECT_SETUP + """
WORK = Path('/kaggle/working/spectrashift-week5-complete')
WORK.mkdir(parents=True, exist_ok=True)
seed_summaries = sorted(INPUT.rglob('week5_seed*_summary.json'))
pilot_summaries = sorted(INPUT.rglob('week5_pilot_summary.json'))
assert len(seed_summaries) == 3 and len(pilot_summaries) == 1
print({'seed_summaries': [str(path) for path in seed_summaries]})
"""
    run = """from spectrashift.train.week5 import aggregate_week5

summary = aggregate_week5(seed_summaries, pilot_summaries[0], WORK / 'week5_run_summary.json')
print(json.dumps(summary, indent=2))
assert summary['week5_complete'] and summary['week6_approved']
assert summary['run_count'] == 45 and summary['evaluation_labels_loaded'] is False
"""
    return _notebook([
        _cell("markdown", "# SpectraShift Week 5: aggregate and approve Week 6\nUse CPU with Internet off. Attach the newest source, Week 5 pilots, and all three Week 5 seed datasets.\n"),
        _cell("code", setup), _cell("code", run),
    ])


def write_notebooks(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "07_week5_prepare.ipynb": prepare_notebook(),
        "08_week5_lr_pilots.ipynb": pilot_notebook(),
        "09a_week5_seed17.ipynb": seed_notebook(17),
        "09b_week5_seed29.ipynb": seed_notebook(29),
        "09c_week5_seed43.ipynb": seed_notebook(43),
        "10_week5_aggregate.ipynb": aggregate_notebook(),
    }
    paths = []
    for name, value in artifacts.items():
        path = output_dir / name
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build output-free Week 5 Kaggle notebooks")
    parser.add_argument("--output-dir", default="notebooks/kaggle")
    args = parser.parse_args()
    for path in write_notebooks(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
