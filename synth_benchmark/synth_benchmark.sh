#!/bin/bash
#SBATCH --job-name=gdsio_bench         # Name of the job
#SBATCH --output=gdsio_bench_%j.out    # Standard output log (%j = Job ID)
#SBATCH --error=gdsio_bench_%j.err     # Standard error log
#SBATCH --partition=c23g                # GPU partition (e.g., 'c18g' or 'gpu')
#SBATCH --gres=gpu:1                   # CRITICAL: Request 4 GPUs for device_sweep [0, 1, 2, 3]
#SBATCH --cpus-per-task=1           # High CPU count to feed the 512-thread POSIX tests
#SBATCH --mem=64G                      # Memory for standard page-cache bounce buffers
#SBATCH --time=00:10:00                # Max runtime (Sweep takes ~70 mins + file init time)
#SBATCH --account=thes2292      # TODO: Replace with your actual project account ID
#SBATCH --beeond


# sbatch synth_benchmark.sh \
# watch squeue -u $USER
# r_wlm_usage -q -p thes2292

module purge
module load CUDA/12.3.0 # TODO: Adjust to the CUDA version used on 

chmod +x synth_benchmark.sh
chmod +x synth_benchmark.py

python -m venv venv
source venv/bin/activate
pip install pandas

srun python synth_benchmark.py

echo "------------------------------------------------------------"
echo "🎉 Job completed! Check your home directory for the CSV files."
echo "------------------------------------------------------------"