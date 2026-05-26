#!/usr/bin/env bash
#SBATCH --job-name=cf3_lerf_full
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --output=/work/runyi_yang/BMVC_2026_FrugaLGS/CF3/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/work/runyi_yang/BMVC_2026_FrugaLGS/CF3/outputs/slurm/%x_%A_%a.err

set -euo pipefail

SCENES=(figurines ramen waldo_kitchen teatime)
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"

# Full, self-contained defaults. Override with --export only when intentionally
# reusing an existing run directory.
REPO_ROOT="/work/runyi_yang/BMVC_2026_FrugaLGS/CF3"
DATA_ROOT="/work/mengjiao_ma/lerf_data/lerf_ovs"
BASE_ROOT="/work/mengjiao_ma/lerf_3dgs"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/lerf_ovs_full_no_vq_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
INIT_ITER=30000
TRAIN_ITERS=3000
FINAL_ITER=$((INIT_ITER + TRAIN_ITERS))
TRAIN_RESOLUTION=2
MASK_THRESH=0.4

OUT_DIR="${RUN_ROOT}/${SCENE}"
mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"

echo "host=$(hostname)"
echo "scene=${SCENE}"
echo "run_root=${RUN_ROOT}"
echo "out_dir=${OUT_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"

if command -v module >/dev/null 2>&1; then
    eval "$(module bash-hook)" || true
    module load nvidia-cuda-12.9.1 || true
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
fi
conda activate cf3

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

cd "${REPO_ROOT}"

# Intentionally no vector quantization: this script does not call
# vectree/vectree.py or any VQ/codebook stage.
python compact_feature_field.py \
    -s "${DATA_ROOT}/${SCENE}" \
    -m "${BASE_ROOT}/${SCENE}" \
    -f langsplat \
    -o "${OUT_DIR}" \
    --images images \
    --iteration "${INIT_ITER}" \
    --iterations "${TRAIN_ITERS}" \
    --resolution "${TRAIN_RESOLUTION}" \
    --eval

python tools/evaluate_lerf_ovs_cf3.py \
    -s "${DATA_ROOT}/${SCENE}" \
    -m "${OUT_DIR}" \
    -f langsplat \
    --images images \
    --iteration "${FINAL_ITER}" \
    --scene "${SCENE}" \
    --gt_folder "${DATA_ROOT}/label/${SCENE}" \
    --mask_thresh "${MASK_THRESH}"

(
    flock -x 9
    python tools/aggregate_lerf_ovs_metrics.py \
        --run_root "${RUN_ROOT}" \
        --scenes figurines ramen waldo_kitchen teatime \
        --output "${RUN_ROOT}/metrics.tsv"
) 9>"${RUN_ROOT}/metrics.lock"

echo "done scene=${SCENE}"
