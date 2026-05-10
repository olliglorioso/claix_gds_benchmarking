#!/bin/bash
#SBATCH --job-name=pt_gds_sweep        # Name of the job
#SBATCH --output=pt_gds_sweep_%j.out   # Standard output log (%j gets replaced with the Job ID)
#SBATCH --error=pt_gds_sweep_%j.err    # Standard error log
#SBATCH --partition=gpu                # Ask for the GPU partition (adjust to 'c18g' or similar if needed)
#SBATCH --gres=gpu:1                   # Request exactly 1 physical GPU
#SBATCH --cpus-per-task=8              # Request 8 CPU cores (good for standard I/O baseline)
#SBATCH --mem=64G                      # Request 64 GB of RAM (needed for the 30GB standard load test)
#SBATCH --time=02:00:00                # Maximum run time (HH:MM:SS)
#SBATCH --account=your_project_id      # TODO: Replace with your actual project account ID

# ==============================================================================
# 1. Environment Setup
# ==============================================================================
echo "Starting job on node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# Purge existing modules and load the exact CUDA version your PyTorch uses
module purge
# TODO: Match this to the CUDA version used to compile your PyTorch (e.g., 12.1 or 12.4)
module load CUDA/12.3.0          

# Activate your Python environment (conda or venv)
# TODO: Uncomment and adjust the path to your environment
# source /home/ts106370/your_env/bin/activate

# ==============================================================================
# 2. Storage Setup (BeeOND)
# ==============================================================================
# Ensure the BEEOND variable is set. On many clusters, requesting a node with 
# local NVMe automatically mounts it and sets $BEEOND. If your cluster uses a 
# different variable for the high-speed scratch (like $TMPDIR or $HPCWORK), 
# map it here so the Python script finds it.

if [ -z "$BEEOND" ]; then
    echo "⚠️ Warning: \$BEEOND is not set by the system."
    # stop here
    exit 1
else
    echo "✅ BeeOND storage detected at: $BEEOND"
fi

source $HOME/claix_gds_benchmarking/claix_gds_benchmarking/bin/activate
python install -r $HOME/claix_gds_benchmarking/claix_gds_benchmarking/requirements.txt
echo "Starting benchmark on node: $HOSTNAME"
srun python $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/ml_benchmark.py