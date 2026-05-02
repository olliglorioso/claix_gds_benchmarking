# gdsio quick start

This repo uses:

- `run_all_gdsio.py`: a Python runner that executes existing `.gdsio` config files, one case at a time, with `iostat` and `gds_stat` collection
- `gdsio_cases/main/`: the 48-case workload matrix
- `gdsio_cases/topology/`: the 16-case per-GPU topology subset

Each benchmark case is a separate `.gdsio` file. The runner does not generate temporary configs.

## 1) Prerequisites

- NVIDIA driver + CUDA + GPUDirect Storage installed
- `gdsio` available, usually at `/usr/local/cuda/gds/tools/gdsio`
- A writable test directory on your target storage, for example `/mnt/nvme/gds_dir`
- Correct GPU index from `nvidia-smi`
- `BEEOND` set in your environment

## 2) Benchmark matrix

The case files are split into two suites:

- `main`: all 4 GPUs active together
- `topology`: one GPU active at a time for path comparison

The `main` suite contains:

- 4 patterns: `seqread`, `seqwrite`, `randread`, `randwrite`
- 2 transports: `gds`, `nogds`
- 3 block sizes: `4K`, `1M`, `4M`
- 2 sizes: `4G`, `32G`
- GPU mode: `all`

That gives `4 x 2 x 3 x 2 = 48` cases.

The `topology` suite contains:

- 2 patterns: `seqread`, `randread`
- 1 transport: `gds`
- 2 block sizes: `4K`, `1M`
- 1 size: `32G`
- 4 individual GPUs: `0`, `1`, `2`, `3`

That gives `2 x 1 x 2 x 1 x 4 = 16` cases.

## 3) Run cases

```bash
python3 run_all_gdsio.py
python3 run_all_gdsio.py --suite main
python3 run_all_gdsio.py --suite topology
```

The runner discovers `.gdsio` files in `gdsio_cases/main/` and `gdsio_cases/topology/`.

## 4) Results

Each run writes raw per-case logs into `logs/`:

- `<case>.<timestamp>.gdsio.log`
- `<case>.<timestamp>.iostat.log`
- `<case>.<timestamp>.gds_stat.log`

Nothing is converted or summarized by the runner. Use the raw tool output later for throughput, CPU utilization, latency, and IOPS analysis.

## 5) Paper-aligned runs

Run the full 48-case workload matrix from Table 1:

```bash
python3 run_all_gdsio.py --suite main
```

Run the 16-case per-GPU topology subset:

```bash
python3 run_all_gdsio.py --suite topology
```

The script uses these built-in defaults:

- `gdsio` binary: `/usr/local/cuda/gds/tools/gdsio`
- case directory: `gdsio_cases`
- log directory: `logs`
- `iostat` command: `iostat 1`
- `gds_stat` command: tries `gds_stats -l 1`, then `gds_stat -l 1`

## 6) Edit the suite

Edit, add, or remove `.gdsio` files under [gdsio_cases](/Users/olliglorioso/Documents/CTHPC/gdsio_cases). The filename stem becomes the case ID used in log filenames.
