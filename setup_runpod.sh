#!/bin/bash
# setup_runpod.sh
# Gemmina Intelligence LLC / Pure Information Laboratory
# AIES 2026 — Phase 6 Expansion: RunPod Environment Setup
#
# Usage:
#   bash setup_runpod.sh
#
# Run this first at the start of Day 2 and Day 3 on RunPod.
# Installs dependencies and pre-downloads both models.
#
# RunPod confirmed settings:
#   GPU:      RTX 4090 / L4 (sm_89 or below — DO NOT use Blackwell/sm_120)
#   Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

set -e

echo "============================================================"
echo "  Phase 6 RunPod Setup — Gemmina Intelligence LLC"
echo "============================================================"
echo ""

# ── 1. GPU Confirmation ──────────────────────────────────────
echo "[1/4] GPU Check..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# Safety check: abort if Blackwell (sm_120) is detected
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
MAJOR=$(echo $COMPUTE_CAP | cut -d'.' -f1)
if [ "$MAJOR" -ge "12" ]; then
    echo "[ERROR] Detected sm_${COMPUTE_CAP} — Blackwell GPU is NOT supported."
    echo "        Required: sm_89 or below (RTX 4090 / L4)."
    echo "        Please terminate this pod and provision a compatible GPU."
    exit 1
fi
echo "[OK] GPU compute capability: sm_${COMPUTE_CAP}"
echo ""

# ── 2. pip Install ───────────────────────────────────────────
echo "[2/4] Installing dependencies..."
pip install -q --upgrade pip
pip install -q \
    "transformers==4.44.2" \
    "accelerate" \
    "scikit-learn" \
    "matplotlib" \
    "scipy" \
    "tqdm" \
    "numpy"
echo "[OK] Dependencies installed."
echo ""

# ── 3. Workspace Setup ───────────────────────────────────────
echo "[3/4] Creating workspace directories..."
mkdir -p /workspace/data/phase6_stimuli
mkdir -p /workspace/data/phase6_results
mkdir -p /workspace/paper_figures
echo "[OK] Directories ready:"
echo "       /workspace/data/phase6_stimuli   <- stimuli JSON"
echo "       /workspace/data/phase6_results   <- experiment outputs"
echo "       /workspace/paper_figures         <- figures (Day 4)"
echo ""

# ── 4. Model Pre-download ────────────────────────────────────
echo "[4/4] Pre-downloading models (this takes ~10-20 min)..."
echo "      Requires HuggingFace token for gated models."
echo ""

# Set your HF token here or export HF_TOKEN before running
if [ -z "$HF_TOKEN" ]; then
    echo "[Warning] HF_TOKEN not set. Gated models (Llama-3) will fail."
    echo "          Export it first: export HF_TOKEN=hf_xxxxxxxxxxxx"
fi

python3 - <<'PYEOF'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

hf_token = os.environ.get("HF_TOKEN", None)

models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

for model_id in models:
    print(f"\n  Downloading: {model_id}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        print(f"    [OK] Tokenizer loaded.")
        # Download weights only (do not load to GPU yet)
        _ = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="cpu",      # CPU only for pre-download
            attn_implementation="eager",
        )
        print(f"    [OK] Weights cached.")
        del _
    except Exception as e:
        print(f"    [ERROR] {model_id}: {e}")
PYEOF

echo ""
echo "============================================================"
echo "  Setup Complete. Ready for Phase 6 experiments."
echo ""
echo "  Next steps:"
echo "  Day 2:"
echo "    python3 phase6_ccr_full.py --build_stimuli --exp A"
echo "    python3 phase6_ccr_full.py --exp B"
echo ""
echo "  Day 3:"
echo "    python3 phase6_ccr_full.py --exp C"
echo "    python3 phase6_ccr_full.py --exp D"
echo "============================================================"
