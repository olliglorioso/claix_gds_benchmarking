#!/bin/bash
#SBATCH --job-name=gdsio_bench         # Name of the job
#SBATCH --output=gdsio_bench_%j.out    # Standard output log (%j = Job ID)
#SBATCH --error=gdsio_bench_%j.err     # Standard error log
#SBATCH --partition=gpu                # GPU partition (e.g., 'c18g' or 'gpu')
#SBATCH --gres=gpu:4                   # CRITICAL: Request 4 GPUs for device_sweep [0, 1, 2, 3]
#SBATCH --cpus-per-task=32             # High CPU count to feed the 512-thread POSIX tests
#SBATCH --mem=64G                      # Memory for standard page-cache bounce buffers
#SBATCH --time=02:30:00                # Max runtime (Sweep takes ~70 mins + file init time)
#SBATCH --account=your_project_id      # TODO: Replace with your actual project account ID
#SBATCH --beeond


module purge
module load CUDA/12.3.0 # TODO: Adjust to the CUDA version used on 
if [ -z "$BEEOND" ]; then
    echo "⚠️ Warning: \$BEEOND is not set by the SLURM environment."
    echo "Falling back to /tmp for storage."
    export BEEOND="/tmp" 
else
    echo "✅ BeeOND high-speed scratch detected at: $BEEOND"
fi

# ==============================================================================
# 3. Execution
# ==============================================================================
echo "------------------------------------------------------------"
echo "🚀 Starting GDSIO Sweep..."
echo "------------------------------------------------------------"
source $HOME/claix_gds_benchmarking/claix_gds_benchmarking/bin/activate
python install -r $HOME/claix_gds_benchmarking/claix_gds_benchmarking/requirements.txt
# Execute the Python benchmark script
srun python synth_benchmark.py

echo "------------------------------------------------------------"
echo "🎉 Job completed! Check your home directory for the CSV files."
echo "------------------------------------------------------------"