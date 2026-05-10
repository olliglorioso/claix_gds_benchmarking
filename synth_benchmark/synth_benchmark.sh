#!/bin/bash
#SBATCH --job-name=gds_benchmark
#SBATCH --output=gds_benchmark_%j.out
#SBATCH --error=gds_benchmark_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH -A <YOUR_PROJECT_ID>

module purge


python .py