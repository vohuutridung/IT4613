#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="/kaggle/working/mind15"
DRY_RUN=0
ALLOW_CPU=0
MODE=""

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --mode)
      ((i += 1))
      MODE="${args[$i]}"
      ;;
    --work-root)
      ((i += 1))
      WORK_ROOT="${args[$i]}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --allow-cpu)
      ALLOW_CPU=1
      ;;
  esac
done

if [[ "$DRY_RUN" == "1" ]]; then
  exec python3 "$ROOT/deployment/kaggle/runner.py" "$@"
fi

mkdir -p "$WORK_ROOT"
if [[ "$ALLOW_CPU" != "1" && "$MODE" =~ ^(train|train-ensemble|single|submit|ensemble-submit|benchmark)$ ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L; then
    echo "No NVIDIA GPU is visible. Select a Kaggle GPU accelerator or pass --allow-cpu." >&2
    exit 1
  fi
fi

UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  python3 -m pip install --user --disable-pip-version-check uv
  UV="$(python3 -m site --user-base)/bin/uv"
fi

REQUIREMENTS="$ROOT/deployment/kaggle/requirements.txt"
VENV="$WORK_ROOT/python311-pytorch29"
TORCH_INDEX=""
if [[ "$MODE" != "prepare" && "$(uname -s)" == "Linux" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TORCH_INDEX="cu126"
  else
    TORCH_INDEX="cpu"
  fi
  VENV="$VENV-$TORCH_INDEX"
fi
if [[ "$MODE" == "prepare" ]]; then
  REQUIREMENTS="$ROOT/deployment/kaggle/requirements-prepare.txt"
  VENV="$WORK_ROOT/python311-prepare"
fi
PYTHON="$VENV/bin/python"
export UV_CACHE_DIR="$WORK_ROOT/.uv-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x "$PYTHON" ]]; then
  "$UV" python install 3.11.15
  "$UV" venv --python 3.11.15 "$VENV"
fi

REQ_HASH="$(shasum -a 256 "$REQUIREMENTS" | awk '{print $1}')"
REQ_MARKER="$VENV/.mind15-requirements-$REQ_HASH"
if [[ ! -f "$REQ_MARKER" ]]; then
  if [[ -n "$TORCH_INDEX" ]]; then
    # Resolve torch from its own index, then install the shared requirements.
    "$UV" pip install --python "$PYTHON" \
      --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" \
      "$(awk '/^torch==/ {print; exit}' "$REQUIREMENTS")"
  fi
  "$UV" pip install --python "$PYTHON" -r "$REQUIREMENTS"
  rm -f "$VENV"/.mind15-requirements-*
  touch "$REQ_MARKER"
  if [[ "$MODE" == "prepare" ]]; then
    "$UV" cache clean
  fi
fi

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
# Respect an explicit device restriction; otherwise expose both Kaggle T4s.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

LOG="$WORK_ROOT/pipeline.log"
set +e
"$PYTHON" "$ROOT/deployment/kaggle/runner.py" "$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" == "0" && "$MODE" == "prepare" ]]; then
  rm -rf "$VENV" "$UV_CACHE_DIR" "$WORK_ROOT/tools/GloVe"
fi
exit "$status"
