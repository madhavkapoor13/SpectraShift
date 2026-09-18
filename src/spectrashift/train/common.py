from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    """Hash a directory independently of mtimes and host-specific metadata."""
    root = Path(path)
    digest = hashlib.sha256()
    for candidate in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = candidate.relative_to(root).as_posix()
        if relative.startswith(".git/") or "__pycache__" in candidate.parts:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_tree_sha256(root: str | Path = ".") -> str:
    root = Path(root)
    digest = hashlib.sha256()
    files = [
        path
        for directory in ("src", "scripts", "configs")
        for path in (root / directory).rglob("*")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"} and "__pycache__" not in path.parts
    ]
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance = Path(root) / "SOURCE_COMMIT"
        if provenance.is_file():
            value = provenance.read_text().strip()
            if len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower()):
                return value
        return "uncommitted"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def select_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        architecture = f"sm_{major}{minor}"
        supported = torch.cuda.get_arch_list()
        if supported and architecture not in supported:
            raise RuntimeError(
                f"{torch.cuda.get_device_name(0)} ({architecture}) is unsupported by this "
                f"PyTorch build ({supported}); select a T4 GPU"
            )
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("This run requires a CUDA GPU")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_record(device: torch.device) -> dict[str, object]:
    result = {
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        result.update({
            "gpu": torch.cuda.get_device_name(0),
            "device_arch": f"sm_{major}{minor}",
            "gpu_count": torch.cuda.device_count(),
        })
    return result


def atomic_torch_save(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: str | Path, record: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
