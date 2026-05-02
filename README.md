# gdsio quick start

This repo now uses:

- `run_all_gdsio.py`: one Python runner with the full benchmark matrix embedded directly in the script
- `gdsio_template.gdsio`: a readable template showing the generated config format

The runner generates temporary `.gdsio` configs on the fly, so there is no separate config file for each test and no CSV to parse.

## 1) Prerequisites

- NVIDIA driver + CUDA + GPUDirect Storage installed
- `gdsio` available, usually at `/usr/local/cuda/gds/tools/gdsio`
- A writable test directory on your target storage, for example `/mnt/nvme/gds_dir`
- Correct GPU index from `nvidia-smi`
- `BEEOND` set in your environment

## 2) Benchmark matrix

The script contains two suites:

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

## 3) List cases

```bash
python3 run_all_gdsio.py --list
python3 run_all_gdsio.py --list --suite main
python3 run_all_gdsio.py --list --suite topology
```

## 4) Run cases

```bash
python3 run_all_gdsio.py
python3 run_all_gdsio.py --suite main
python3 run_all_gdsio.py --suite topology
```

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
- template file: `gdsio_template.gdsio`
- log directory: `logs`
- runtime: `60`
- `do_verify`: `0`

## 6) Edit the suite

Edit the `CASES` array in [run_all_gdsio.py](/Users/olliglorioso/Documents/CTHPC/run_all_gdsio.py).
