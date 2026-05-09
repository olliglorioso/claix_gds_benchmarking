import os
import time
import torch
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def

# --- Configuration ---
if 'DALI_EXTRA_PATH' not in os.environ:
    raise RuntimeError("DALI_EXTRA_PATH is not set.")

#DATA_DIR = os.path.join(
#    os.environ['DALI_EXTRA_PATH'], 
#    'db', '3D', 'MRI', 'Knee', 'npy_2d_slices', 'STU00001'
#)
DATA_DIR = os.path.join('/hpcwork', 'ts106370', 'massive_gds_dataset')
# --- 1. Pipeline Definitions ---

def build_standard_pipeline(batch_size, num_threads):
    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def pipe_standard():
        # Read to CPU RAM, then transfer to GPU
        data = fn.readers.numpy(device='cpu', file_root=DATA_DIR)
        return data.gpu()
    return pipe_standard()

def build_gds_pipeline(batch_size, num_threads):
    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def pipe_gds():
        # Direct NVMe to GPU VRAM via libcufile
        data = fn.readers.numpy(device='gpu', file_root=DATA_DIR, use_o_direct=True)
        return data
    return pipe_gds()

# --- 2. The Benchmarking Engine ---

def run_benchmark(pipeline, name, iterations=50):
    print(f"\n{'='*40}")
    print(f"Benchmarking: {name}")
    print(f"{'='*40}")
    
    # 1. Build the pipeline
    pipeline.build()

    # 2. Warmup (Discard results)
    print("Warming up (5 iterations)...")
    for _ in range(5):
        _ = pipeline.run()

    # Calculate data size dynamically from one output
    out = pipeline.run()
    sample_data = out[0].as_cpu().as_array()
    bytes_per_batch = sample_data.nbytes
    
    # 3. Timed Benchmark Loop
    print(f"Running {iterations} timed iterations...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(iterations):
        _ = pipeline.run()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 4. Calculate Metrics
    total_time = end_time - start_time
    total_bytes_transferred = bytes_per_batch * iterations
    bandwidth_gbs = (total_bytes_transferred / total_time) / (1024**3)

    print("-" * 40)
    print(f"Total Time:      {total_time:.4f} seconds")
    print(f"Data Processed:  {total_bytes_transferred / (1024**3):.2f} GB")
    print(f"Throughput:      {bandwidth_gbs:.2f} GB/s")
    print("=" * 40)
    
    return bandwidth_gbs

# --- Main Execution ---
if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_THREADS = 4
    ITERATIONS = 100


    # Run Standard (CPU Bounce Buffer)
    pipe_std = build_standard_pipeline(BATCH_SIZE, NUM_THREADS)
    std_bw = run_benchmark(pipe_std, "Standard (CPU -> GPU)", ITERATIONS)

    # Free memory before running the next test
    del pipe_std
    torch.cuda.empty_cache()

    # Run GDS (NVMe -> GPU)
    pipe_gds = build_gds_pipeline(BATCH_SIZE, NUM_THREADS)
    gds_bw = run_benchmark(pipe_gds, "GPUDirect Storage (GDS)", ITERATIONS)

    # Print Summary
    print(f"\nFinal Result: GDS was {gds_bw / std_bw:.2f}x faster than Standard I/O.")