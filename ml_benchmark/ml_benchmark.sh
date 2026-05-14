#!/bin/bash
#SBATCH --job-name=ml_bench         # Name of the job
#SBATCH --output=mlbench__%j.out    # Standard output log (%j = Job ID)
#SBATCH --error=mlbench__%j.err     # Standard error log
#SBATCH --partition=c23g                # GPU partition (e.g., 'c18g' or 'gpu')
#SBATCH --gres=gpu:1                   # CRITICAL: Request 4 GPUs for device_sweep [0, 1, 2, 3]
#SBATCH --cpus-per-task=1           # High CPU count to feed the 512-thread POSIX tests
#SBATCH --mem=64G                      # Memory for standard page-cache bounce buffers
#SBATCH --time=00:10:00                # Max runtime (Sweep takes ~70 mins + file init time)
#SBATCH --account=thes2292      # TODO: Replace with your actual project account ID
#SBATCH --beeond

# ==============================================================================
# 1. Environment Setup
# ==============================================================================
echo "Starting job on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
module load CUDA/12.3.0          

chmod +x ml_benchmark.sh
chmod +x ml_benchmark.py

python -m venv venv
source venv/bin/activate
pip install pandas torch 

if [ -z "$BEEOND" ]; then
    echo "⚠️ Warning: \$BEEOND is not set by the system."
    # stop here
    exit 1
else
    echo "✅ BeeOND storage detected at: $BEEOND"
fi

echo "Starting benchmark on node: $HOSTNAME"
srun python $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/ml_benchmark.py