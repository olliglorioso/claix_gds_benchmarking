#!/bin/bash
#SBATCH --job-name=sf_bench         # Name of the job
#SBATCH --output=sf___bench_%j.out    # Standard output log (%j = Job ID)
#SBATCH --error=sf___bench_%j.err     # Standard error log
#SBATCH --partition=c23g                # GPU partition (e.g., 'c18g' or 'gpu')
#SBATCH --gres=gpu:1                   # CRITICAL: Request 4 GPUs for device_sweep [0, 1, 2, 3]
#SBATCH --cpus-per-task=1           # High CPU count to feed the 512-thread POSIX tests
#SBATCH --mem=64G                      # Memory for standard page-cache bounce buffers
#SBATCH --time=00:10:00                # Max runtime (Sweep takes ~70 mins + file init time)
#SBATCH --account=thes2292      # TODO: Replace with your actual project account ID
#SBATCH --beeond

git config --global credential.helper store
cp -r $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/Meta-Llama-Guard-2-8B $BEEOND

# 1. Clean the environment and load the correct NVIDIA drivers
module load CUDA/12.3.0             # Ensure this matches the CUDA version your PyTorch environment expects

chmod +x safetensors_benchmark.sh
chmod +x safetensors_benchmark.py

python -m venv venv
source venv/bin/activate
pip install pandas torch kvikio-cu12 glob tqdm

# 2. Activate your Python environment (uncomment and adjust if you use conda/venv)
# source /rwthfs/rz/cluster/home/ts106370/your_env/bin/activate

export KVIKIO_LOG_LEVEL=TRACE

# 4. Execute the Python script
echo "Starting benchmark on node: $HOSTNAME"
srun python $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/safetensors_benchmark.py