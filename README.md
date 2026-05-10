# HPC with GDS Benchmarking

Project is divided into synthetic and ML benchmarks, which can be run separately.

## Quick start ML
- ML Benchmarks contains practical safetensor loading with Kvikio and synthetic weights loading with PyTorch.

### Safetensors quick start
1. Download the safetensors from Hugging Face:
`git clone https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B $HOME/claix_gds_benchmarking/claix_gds_benchmarking`
2. Ensure that we have a Python venv with the required dependencies:
```bash
mkdir $HOME/claix_gds_benchmarking/claix_gds_benchmarking/st_venv
python3.11 -m venv $HOME/claix_gds_benchmarking/claix_gds_benchmarking/st_venv
source $HOME/claix_gds_benchmarking/claix_gds_benchmarking/st_venv/bin/activate
pip install -r $HOME/claix_gds_benchmarking/claix_gds_benchmarking/requirements.txt
```
3. Run the script (exports required environment variables and runs & logs everything automatically):
`sbatch $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/safetensors_bench.sh`

### Synthetic ML quick start
1. Ensure that we have a Python venv with the required dependencies from the safetensors quick start.
2. Run the script (exports required environment variables and runs & logs everything automatically):
`sbatch $HOME/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/ml_benchmark.sh`