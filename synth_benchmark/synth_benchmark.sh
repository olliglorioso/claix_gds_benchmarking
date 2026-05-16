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

if [ -n "${CLAIX_GDS_BENCHMARK_ROOT:-}" ]; then
    BENCHMARK_ROOT="${CLAIX_GDS_BENCHMARK_ROOT}"
elif [ -f "${HOME}/claix_gds_benchmarking/claix_gds_benchmarking/requirements.txt" ]; then
    BENCHMARK_ROOT="${HOME}/claix_gds_benchmarking/claix_gds_benchmarking"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/requirements.txt" ]; then
    BENCHMARK_ROOT="${SLURM_SUBMIT_DIR}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/claix_gds_benchmarking/requirements.txt" ]; then
    BENCHMARK_ROOT="${SLURM_SUBMIT_DIR}/claix_gds_benchmarking"
else
    echo "Error: could not locate claix_gds_benchmarking. Set CLAIX_GDS_BENCHMARK_ROOT to the package directory."
    exit 1
fi

SCRIPT_DIR="${BENCHMARK_ROOT}/synth_benchmark"
VENV_DIR="${SCRIPT_DIR}/.venv-py311"

PYTHON_BIN=""
for candidate in "${CLAIX_GDS_PYTHON:-}" python3.12 python3.11 python3; do
    if [ -z "${candidate}" ] || ! command -v "${candidate}" >/dev/null 2>&1; then
        continue
    fi
    if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    echo "Error: Python 3.11 or newer is required. Set CLAIX_GDS_PYTHON=/path/to/python3.11 if it is not on PATH."
    exit 1
fi

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
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Error: existing venv at ${VENV_DIR} is not Python 3.11+. Remove it or set CLAIX_GDS_PYTHON=/path/to/python3.11."
    exit 1
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install pandas

echo "BeeOND storage: ${BEEOND}"
echo "gdsio binary: ${GDSIO_PATH}"
echo "Running synthetic benchmark..."
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
    "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/synth_benchmark.py" \
    --gdsio-path "${GDSIO_PATH}" \
    --output-dir "${BEEOND}" \
    --results-dir "${SCRIPT_DIR}"

echo "Synthetic benchmark completed. CSV files are in ${SCRIPT_DIR}."
