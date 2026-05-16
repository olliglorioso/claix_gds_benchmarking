#!/bin/bash
#SBATCH --job-name=safetensors_bench
#SBATCH --chdir=/home/ts106370/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark
#SBATCH --output=safetensors_benchmark_%j.out
#SBATCH --error=safetensors_benchmark_%j.err
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --account=thes2292
#SBATCH --beeond

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/Meta-Llama-Guard-2-8B"
VENV_DIR="${SCRIPT_DIR}/.venv"

cd "${SCRIPT_DIR}"

echo "Starting safetensors benchmark"
echo "Node: ${HOSTNAME}"
echo "Job ID: ${SLURM_JOB_ID:-not-a-slurm-job}"
echo "Work directory: ${SCRIPT_DIR}"

module purge
module load CUDA/12.3.0

if [ -z "${BEEOND:-}" ]; then
    echo "Error: BEEOND is not set. Submit this script with #SBATCH --beeond."
    exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: model directory not found: ${MODEL_DIR}"
    exit 1
fi

echo "BeeOND storage: ${BEEOND}"
echo "Copying safetensors model to BeeOND..."
cp -a "${MODEL_DIR}" "${BEEOND}/"

if [ ! -d "${VENV_DIR}" ]; then
    python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install pandas torch kvikio-cu12==26.4.0 tqdm

export KVIKIO_LOG_LEVEL="${KVIKIO_LOG_LEVEL:-WARN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Running safetensors benchmark..."
srun python "${SCRIPT_DIR}/safetensors_benchmark.py"
