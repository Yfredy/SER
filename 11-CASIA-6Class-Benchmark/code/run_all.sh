#!/usr/bin/env bash
# One-shot reproduction of the SER three-model / four-corpus benchmark.
#
# Pipeline:
#   1. run_torch.py     -> ert_wfccl.json     (torch env)
#   2. run_tf_light.py  -> light_sernet.json  (tf env)
#   3. run_tf_tim.py    -> tim_net.json       (tf env)
#   4. render.py        -> merged.json + docs/ser-benchmark-comparison.md (torch env)
#
# TF_USE_LEGACY_KERAS=1 is mandatory before both TensorFlow steps: TIM-Net's
# official TIMNET layer calls K.reverse on a symbolic tensor, which Keras 3
# rejects.
#
# Idempotent: each runner overwrites its own result JSON, and render.py
# rebuilds merged.json + the document from scratch. Logs land in
# benchmark_ser/results/logs/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TORCH_PYTHON="${TORCH_PYTHON:-/home/bytedance/miniconda3/envs/torch-cu124/bin/python}"
TF_PYTHON="${TF_PYTHON:-/home/bytedance/miniconda3/envs/tf-gpu/bin/python}"

LOGDIR="benchmark_ser/results/logs"
mkdir -p "$LOGDIR"

TS="$(date +%Y%m%d-%H%M%S)"

# ---------------------------------------------------------------------------
# Step 1: ERT-WFCCL (PyTorch)
# ---------------------------------------------------------------------------
echo "[run_all] Step 1/4: ERT-WFCCL (torch env)" | tee "$LOGDIR/run_all-$TS.log"
"$TORCH_PYTHON" -m benchmark_ser.run_torch \
    --out benchmark_ser/results/ert_wfccl.json \
    2>&1 | tee -a "$LOGDIR/run_all-$TS.log"

# ---------------------------------------------------------------------------
# Step 2: Light-SERNet (TensorFlow)
# ---------------------------------------------------------------------------
echo "[run_all] Step 2/4: Light-SERNet (tf env)" | tee -a "$LOGDIR/run_all-$TS.log"
export TF_USE_LEGACY_KERAS=1
"$TF_PYTHON" -m benchmark_ser.run_tf_light \
    --out benchmark_ser/results/light_sernet.json \
    2>&1 | tee -a "$LOGDIR/run_all-$TS.log"

# ---------------------------------------------------------------------------
# Step 3: TIM-Net (TensorFlow)
# ---------------------------------------------------------------------------
echo "[run_all] Step 3/4: TIM-Net (tf env)" | tee -a "$LOGDIR/run_all-$TS.log"
export TF_USE_LEGACY_KERAS=1
"$TF_PYTHON" -m benchmark_ser.run_tf_tim \
    --out benchmark_ser/results/tim_net.json \
    2>&1 | tee -a "$LOGDIR/run_all-$TS.log"

# ---------------------------------------------------------------------------
# Step 4: merge + render (PyTorch env; numpy + statistics only)
# ---------------------------------------------------------------------------
echo "[run_all] Step 4/4: merge + render (torch env)" | tee -a "$LOGDIR/run_all-$TS.log"
"$TORCH_PYTHON" -m benchmark_ser.render \
    2>&1 | tee -a "$LOGDIR/run_all-$TS.log"

echo "[run_all] done. Log: $LOGDIR/run_all-$TS.log"
