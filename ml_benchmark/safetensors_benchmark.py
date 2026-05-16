import os
import glob
import torch
import json
import struct
import time
import csv
from kvikio.cufile import CuFile
from kvikio import cufile_driver
from kvikio import defaults

# --- 1. Safetensors to PyTorch Dtype Mapping ---
SAFETENSORS_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I8":  torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
}

def load_safetensor_with_gds(filepath):
    """Loads a single safetensors file directly to VRAM via GDS."""
    with open(filepath, 'rb') as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        
    data_start_offset = 8 + header_size
    tensors = {}
    f_gds = CuFile(filepath, "r")
    
    for tensor_name, metadata in header.items():
        if tensor_name == "__metadata__":
            continue
            
        shape = metadata['shape']
        st_dtype = metadata['dtype']
        pt_dtype = SAFETENSORS_DTYPE_MAP.get(st_dtype, torch.float32)
        
        # Allocate pinned VRAM
        gpu_tensor = torch.empty(shape, dtype=pt_dtype, device='cuda')
        
        start_byte = metadata['data_offsets'][0]
        end_byte = metadata['data_offsets'][1]
        byte_length = end_byte - start_byte
        file_offset = data_start_offset + start_byte
        
        # NVMe -> PCIe -> VRAM (GDS DMA)
        f_gds.read(gpu_tensor, size=byte_length, file_offset=file_offset)
        tensors[tensor_name] = gpu_tensor
        
    f_gds.close()
    return tensors

def run_benchmark(filepaths, method="GDS", iterations=10, task_size=(4*1024*1024)):
    print(f"\n{'-'*50}")
    print(f"🚀 Starting Benchmark: {method} | Task Size: {task_size/(1024*1024):.2f}M ({iterations} iterations)")
    print(f"{'-'*50}")
    
    # --- ADD THIS TOGGLE ---
    if method == "Standard (CPU-Bounce)":
        defaults.set({
            "compat_mode": 1, 
            "auto_direct_io_read": 0,
            "num_threads": 8,
            "task_size": task_size
        })
    else:
        defaults.set({
            "compat_mode": 0, 
            "auto_direct_io_read": 1,
            "num_threads": 8,
            "task_size": task_size
        })
        
    total_bytes = sum(os.path.getsize(fp) for fp in filepaths)
    total_gb = total_bytes / (1024**3)
    metrics = []

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...", end="", flush=True)
        full_state_dict = {}
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        
        for fp in filepaths:
            # Both methods now use KvikIO. The environment variable 
            # dictates whether it uses DMA or the CPU bounce buffer.
            shard_tensors = load_safetensor_with_gds(fp)
            full_state_dict.update(shard_tensors)
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        throughput = total_gb / latency
        metrics.append({
            "Iteration": i + 1,
            "Method": method,
            "Task_Size_Bytes": task_size,  # <-- Added to metrics
            "Data_Size_GB": round(total_gb, 4),
            "Latency_sec": round(latency, 4),
            "Throughput_GB_s": round(throughput, 4)
        })
        
        print(f" Latency: {latency:.2f}s | Throughput: {throughput:.2f} GB/s")
        
        # Free memory immediately to prevent OOM
        del full_state_dict

    return metrics

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Health Check
    try:
        gds_avail = cufile_driver.get("is_gds_available")
        print(f"Is gds available: {gds_avail}")
        
        compat_mode = cufile_driver.get("allow_compat_mode")
        print(f"Is compatibility mode avail: {compat_mode}")
    except Exception as e:
        print(f"⚠️ Warning: Could not read properties. Error: {e}")

    # 2. Locate Data
    raw_path = "$BEEOND/Meta-Llama-Guard-2-8B/*.safetensors"
    # dev 
    # raw_path = "~/claix_gds_benchmarking/claix_gds_benchmarking/ml_benchmark/Meta-Llama-Guard-2-8B/*.safetensors"
    
    pathed = os.path.expanduser(os.path.expandvars(raw_path))
    filepaths = sorted(glob.glob(pathed))
    
    if not filepaths:
        raise FileNotFoundError(f"No files found at: {pathed}")
        
    print(f"\nFound {len(filepaths)} safetensors files.")

    # 3. Define the Sweep Parameters
    task_sizes = [
        64 * 1024,        # 64K
        256 * 1024,       # 256K
        1 * 1024 * 1024,  # 1M
        4 * 1024 * 1024,  # 4M
        16 * 1024 * 1024, # 16M
        64 * 1024 * 1024  # 64M
    ]
    ITERATIONS = 10
    all_results = []
    
    # 4. Run Benchmark Sweep
    for ts in task_sizes:
        print(f"\n{'='*60}")
        print(f"🧪 SWEEPING TASK SIZE: {ts / (1024*1024):.2f} MB")
        print(f"{'='*60}")
        
        # Benchmark 1: Standard POSIX / CPU Bounce Buffer
        std_results = run_benchmark(filepaths, method="Standard (CPU-Bounce)", iterations=ITERATIONS, task_size=ts)
        all_results.extend(std_results)
        
        # Benchmark 2: GPUDirect Storage (GDS)
        gds_results = run_benchmark(filepaths, method="GDS", iterations=ITERATIONS, task_size=ts)
        all_results.extend(gds_results)

    # 5. Save to CSV next to this benchmark script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "gds_vs_posix_benchmark.csv")
    
    with open(csv_file, mode="w", newline="") as f:
        # Added "Task_Size_Bytes" to the fieldnames
        writer = csv.DictWriter(f, fieldnames=["Iteration", "Method", "Task_Size_Bytes", "Data_Size_GB", "Latency_sec", "Throughput_GB_s"])
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\n🎉 Sweep complete! All results saved to: {csv_file}")
