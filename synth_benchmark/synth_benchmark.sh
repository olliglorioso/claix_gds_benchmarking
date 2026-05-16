#!/bin/bash
#SBATCH --job-name=gdsio_bench
#SBATCH --chdir=/home/ts106370/claix_gds_benchmarking/claix_gds_benchmarking/synth_benchmark
#SBATCH --output=synth_benchmark_%j.out
#SBATCH --error=synth_benchmark_%j.err
#SBATCH --partition=c23g
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --account=thes2292
#SBATCH --beeond

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

cd "${SCRIPT_DIR}"

echo "Starting synthetic gdsio benchmark"
echo "Node: ${HOSTNAME}"
echo "Job ID: ${SLURM_JOB_ID:-not-a-slurm-job}"
echo "Work directory: ${SCRIPT_DIR}"

module purge
module load CUDA/12.3.0

if [ -z "${BEEOND:-}" ]; then
    echo "Error: BEEOND is not set. Submit this script with #SBATCH --beeond."
    exit 1
fi

GDSIO_PATH="${CUDA_HOME:-/usr/local/cuda}/gds/tools/gdsio"
if [ ! -x "${GDSIO_PATH}" ]; then
    GDSIO_PATH="/usr/local/cuda/gds/tools/gdsio"
fi
if [ ! -x "${GDSIO_PATH}" ]; then
    echo "Error: gdsio not found. Checked CUDA_HOME and /usr/local/cuda/gds/tools/gdsio."
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install pandas

echo "BeeOND storage: ${BEEOND}"
echo "gdsio binary: ${GDSIO_PATH}"
echo "Running synthetic benchmark..."
srun python "${SCRIPT_DIR}/synth_benchmark.py" --gdsio-path "${GDSIO_PATH}" --output-dir "${BEEOND}" --results-dir "${SCRIPT_DIR}"

echo "Synthetic benchmark completed. CSV files are in ${SCRIPT_DIR}."
