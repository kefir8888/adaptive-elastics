#!/usr/bin/env bash
# Bootstrap a rented CUDA box (Vast.ai / RunPod / Lambda) for training.
# Usage: curl -fsSL https://raw.githubusercontent.com/kefir8888/adaptive-elastics/main/scripts/gpu_box_setup.sh | bash
set -euo pipefail

cd "$HOME"
command -v git >/dev/null || (apt-get update && apt-get install -y git)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d adaptive-elastics ] || git clone https://github.com/kefir8888/adaptive-elastics.git
cd adaptive-elastics && git pull
uv sync --extra cuda

uv run python -c "import jax; print('jax backend:', jax.default_backend(), jax.devices())"
echo "Ready. Train with:"
echo "  cd ~/adaptive-elastics && nohup uv run pea-train --config configs/baseline.yaml --output_dir ~/runs > ~/train.log 2>&1 &"
