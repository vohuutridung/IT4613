"""On-disk PyTorch checkpoints shared by training, prediction and Kaggle."""
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


CHECKPOINT_STATE = "checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_NAME = re.compile(r"ckpt-(0|[1-9][0-9]*)\.pt")


def read_checkpoint_name(state):
    name = Path(state).read_text(encoding="utf-8").strip()
    if not CHECKPOINT_NAME.fullmatch(name):
        raise ValueError("checkpoint state must point to a relative PyTorch ckpt-N.pt file")
    return name


def checkpoint_update(path):
    match = CHECKPOINT_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError("invalid PyTorch checkpoint filename")
    return int(match[1])


def latest_checkpoint(run_dir):
    directory = Path(run_dir) / "checkpoints"
    state = directory / CHECKPOINT_STATE
    if not state.is_file():
        return None
    path = directory / read_checkpoint_name(state)
    if not path.is_file():
        raise FileNotFoundError("missing checkpoint file: {}".format(path))
    return path


def atomic_write(path, write):
    temp = NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False)
    pending = Path(temp.name)
    try:
        with temp:
            write(temp)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def save_checkpoint(run_dir, state):
    import torch

    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ckpt-{}.pt".format(state["update"])
    atomic_write(path, lambda stream: torch.save(state, stream))
    atomic_write(directory / CHECKPOINT_STATE,
                 lambda stream: stream.write((path.name + "\n").encode("utf-8")))
    older = sorted((p for p in directory.glob("ckpt-*.pt")
                    if p != path and CHECKPOINT_NAME.fullmatch(p.name)),
                   key=checkpoint_update, reverse=True)
    for stale in older[1:]:
        stale.unlink(missing_ok=True)
    return path


def load_checkpoint(path, config_hash):
    import torch

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if (state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or state.get("config_sha256") != config_hash
            or state.get("update") != checkpoint_update(path)):
        raise ValueError("PyTorch checkpoint schema, run fingerprint, or update counter mismatch")
    return state
