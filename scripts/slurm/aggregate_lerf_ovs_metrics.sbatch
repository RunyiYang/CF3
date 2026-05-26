#!/usr/bin/env bash
#SBATCH --job-name=cf3_lerf_agg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work/runyi_yang/BMVC_2026_FrugaLGS/CF3}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must point to the fresh output directory}"

cd "${REPO_ROOT}"
python tools/aggregate_lerf_ovs_metrics.py \
    --run_root "${RUN_ROOT}" \
    --scenes figurines ramen waldo_kitchen teatime \
    --output "${RUN_ROOT}/metrics.tsv"
