#!/bin/bash
#SBATCH --job-name=gds_safetensors     # Name of the job
#SBATCH --output=gds_log_%j.out        # Standard output log (%j gets replaced with the Job ID)
#SBATCH --error=gds_log_%j.err         # Standard error log
#SBATCH --partition=gpu                # Ask for the GPU partition (check if you need 'c18g' or similar on CLAIX)
#SBATCH --gres=gpu:1                   # Request exactly 1 physical GPU
#SBATCH --cpus-per-task=4              # Request 4 CPU cores
#SBATCH --mem=32G                      # Request 32 GB of system RAM
#SBATCH --time=00:30:00                # Maximum time the job will run (HH:MM:SS)
#SBATCH -A <YOUR_ACC_ID>           # Replace with your actual project ID on CLAIX

# Beeond related stuff

#SBATCH --beeond
git config --global credential.helper store
git clone https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B $BEEOND

# 1. Clean the environment and load the correct NVIDIA drivers
module purge
module load CUDA/12.1   # Ensure this matches the CUDA version your PyTorch environment expects

# 2. Activate your Python environment (uncomment and adjust if you use conda/venv)
# source /rwthfs/rz/cluster/home/ts106370/your_env/bin/activate

export KVIKIO_COMPAT_MODE=ON
export KVIKIO_AUTO_DIRECT_IO_READ=1
export KVIKIO_LOG_LEVEL=TRACE

source $HOME/claix_gds_benchmarking/claix_gds_benchmarking/bin/activate  # Activate your Python environment (adjust path if needed)

# 4. Execute the Python script
echo "Starting benchmark on node: $HOSTNAME"
srun python $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/safetensors_benchmark.py