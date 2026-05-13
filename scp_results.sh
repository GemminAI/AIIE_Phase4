#!/bin/bash
# scp_results.sh
# Gemmina Intelligence LLC / Pure Information Laboratory
# AIES 2026 — Phase 6 Expansion: RunPod Results Download
#
# Usage:
#   bash scp_results.sh <PORT> <IP> <LOCAL_DEST>
#
# Example:
#   bash scp_results.sh 22345 104.xxx.xxx.xxx /Users/tomonam3/vaults/GemminAI/AIIE_Phase4/results/phase6
#
# Run after Day 2 (Exp A/B) and after Day 3 (Exp C/D).

set -e

PORT=${1:?"Usage: bash scp_results.sh <PORT> <IP> <LOCAL_DEST>"}
IP=${2:?"Usage: bash scp_results.sh <PORT> <IP> <LOCAL_DEST>"}
LOCAL_DEST=${3:?"Usage: bash scp_results.sh <PORT> <IP> <LOCAL_DEST>"}

echo "============================================================"
echo "  Phase 6 Results Download — Gemmina Intelligence LLC"
echo "  RunPod: ${IP}:${PORT}"
echo "  Local:  ${LOCAL_DEST}"
echo "============================================================"

mkdir -p "${LOCAL_DEST}"

echo ""
echo "[1/3] Downloading JSON results..."
scp -P "${PORT}" -r \
    "root@${IP}:/workspace/data/phase6_results/*.json" \
    "${LOCAL_DEST}/"

echo ""
echo "[2/3] Downloading figures..."
scp -P "${PORT}" -r \
    "root@${IP}:/workspace/paper_figures/fig_*.png" \
    "${LOCAL_DEST}/" 2>/dev/null || echo "  (No figures yet — skipping)"

echo ""
echo "[3/3] Downloading logs..."
scp -P "${PORT}" -r \
    "root@${IP}:/workspace/*.log" \
    "${LOCAL_DEST}/" 2>/dev/null || echo "  (No logs — skipping)"

echo ""
echo "============================================================"
echo "  Download complete."
echo ""
echo "  Files in ${LOCAL_DEST}:"
ls -lh "${LOCAL_DEST}/"
echo ""
echo "  Next: Run phase6_analyzer.py on downloaded data."
echo "    python3 phase6_analyzer.py \\"
echo "      --data_dir ${LOCAL_DEST} \\"
echo "      --out_dir  <YOUR_FIGURES_DIR>"
echo "============================================================"
