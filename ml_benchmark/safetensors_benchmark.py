import os
import glob
import torch
import json
import struct
import time
import csv
from kvikio.cufile import CuFile
from kvikio import cufile_driver

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
    f_gds.drop_system_page_cache()
    
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

def run_benchmark(filepaths, method="GDS", iterations=10):
    print(f"\n{'-'*50}")
    print(f"🚀 Starting Benchmark: {method} ({iterations} iterations)")
    print(f"{'-'*50}")
    
    # --- ADD THIS TOGGLE ---
    if method == "Standard (CPU-Bounce)":
        os.environ["KVIKIO_COMPAT_MODE"] = "1"
    else:
        os.environ.pop("KVIKIO_COMPAT_MODE", None)
        
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
            # --- UPDATE THIS BLOCK ---
            # Both methods now use KvikIO. The environment variable 
            # dictates whether it uses DMA or the CPU bounce buffer.
            shard_tensors = load_safetensor_with_gds(fp)
            full_state_dict.update(shard_tensors)
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        throughput = total_gb / latency
        # ... (rest of the metric calculation remains exactly the same)

# --- Main Execution ---
if __name__ == "__main__":    
    # 1. Health Check (Using your preferred cufile_driver approach)
    try:
        gds_avail = cufile_driver.get("is_gds_available")
        print(f"Is gds available: {gds_avail}")
        
        compat_mode = cufile_driver.get("allow_compat_mode")
        print(f"Is compatibility mode avail: {compat_mode}")
    except Exception as e:
        print(f"⚠️ Warning: Could not read properties. Error: {e}")

    # 2. Locate Data
    # expandvars resolves $BEEOND, expanduser resolves ~
    raw_path = "$BEEOND/Meta-Llama-Guard-2-8B/*.safetensors"
    pathed = os.path.expanduser(os.path.expandvars(raw_path))
    filepaths = sorted(glob.glob(pathed))
    
    if not filepaths:
        raise FileNotFoundError(f"No files found at: {pathed}")
        
    print(f"\nFound {len(filepaths)} safetensors files.")

    # 3. Run Benchmarks
    ITERATIONS = 10
    all_results = []
    
    # Benchmark 1: Standard POSIX / CPU Bounce Buffer
    std_results = run_benchmark(filepaths, method="Standard (CPU-Bounce)", iterations=ITERATIONS)
    all_results.extend(std_results)
    
    # Benchmark 2: GPUDirect Storage (GDS)
    gds_results = run_benchmark(filepaths, method="GDS", iterations=ITERATIONS)
    all_results.extend(gds_results)

    # 4. Save to CSV in $HOME
    csv_file = os.path.expanduser("~/gds_vs_posix_benchmark.csv")
    
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Iteration", "Method", "Data_Size_GB", "Latency_sec", "Throughput_GB_s"])
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\n🎉 Benchmark complete! Results saved to: {csv_file}")