# gdsio quick start

This repository includes a sample `gdsio` config file (`config.gdsio`) for GPUDirect Storage benchmarking.

## 1) Prerequisites

- NVIDIA driver + CUDA + GPUDirect Storage installed
- `gdsio` available (usually at `/usr/local/cuda/gds/tools/gdsio`)
- A writable test directory on your target storage (example: `/mnt/nvme/gds_dir`)
- Correct GPU index from `nvidia-smi`

## 2) Run with config files

```bash
/usr/local/cuda/gds/tools/gdsio config.write.gdsio
/usr/local/cuda/gds/tools/gdsio config.read.gdsio
/usr/local/cuda/gds/tools/gdsio config.randread.gdsio
```
