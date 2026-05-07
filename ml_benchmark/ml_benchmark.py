import torch
import torch.nn as nn
import time
import os
import gc
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.serialization import skip_data

# ================= CONFIGURATION =================
# Set 4KB alignment required by GDS
torch.utils.serialization.config.save.storage_alignment = 4096

def create_heavy_model(size_gb):
    """Creates a model with roughly the requested size in GB (using FP32)."""
    # 1 parameter (FP32) = 4 bytes. 
    # Size in bytes = size_gb * 1024^3.
    num_params = int((size_gb * (1024**3)) / 4)
    print(f"Creating model with {num_params:,} parameters (~{size_gb} GB)...")
    
    # We use a simple list of large tensors to simulate a state_dict
    model_tensors = {
        f"layer_{i}": torch.randn(num_params // 10, device='cuda') 
        for i in range(10)
    }
    return model_tensors

def get_checkpoint_offsets(path, state_dict):
    """
    Uses FakeTensorMode to identify where each tensor's data 
    is located within the checkpoint file.
    """
    with FakeTensorMode():
        # Loading with FakeTensor doesn't materialize data, 
        # but tags the tensors with their file offsets.
        fake_state_dict = torch.load(path, weights_only=True)
        
    offsets = {}
    for name, tensor in fake_state_dict.items():
        # The prototype API tags FakeTensors with the offset
        if hasattr(tensor, "checkpoint_offset"):
            offsets[name] = tensor.checkpoint_offset
    return offsets

# ================= BENCHMARK FUNCTIONS =================

def benchmark_standard(state_dict, path, iterations):
    print(f"\n--- Benchmarking Standard PyTorch ({iterations} iterations) ---")
    
    # Save Benchmark
    start_save = time.time()
    for _ in range(iterations):
        torch.save(state_dict, path)
    end_save = time.time()
    
    # Load Benchmark
    start_load = time.time()
    for _ in range(iterations):
        _ = torch.load(path, map_location="cuda", weights_only=True)
    end_load = time.time()
    
    return (end_save - start_save) / iterations, (end_load - start_load) / iterations

def benchmark_gds(state_dict, path, iterations):
    print(f"\n--- Benchmarking GDS Prototype ({iterations} iterations) ---")
    
    # 1. PRE-STEP: Reserve space and get offsets (only needed once)
    with skip_data():
        torch.save(state_dict, path)
    offsets = get_checkpoint_offsets(path, state_dict)
    
    # 2. GDS SAVE BENCHMARK
    start_save = time.time()
    for _ in range(iterations):
        # Open GDS File
        gds_file = torch.cuda.gds.GdsFile(path, os.O_WRONLY)
        for name, tensor in state_dict.items():
            # Blast tensor bytes directly to reserved space on NVMe
            gds_file.save_storage(tensor.untyped_storage(), offset=offsets[name])
    end_save = time.time()

    # 3. GDS LOAD BENCHMARK
    start_load = time.time()
    for _ in range(iterations):
        # Load metadata/empty tensors first
        with skip_data():
            loaded_dict = torch.load(path, weights_only=True)
        
        # Fill empty tensors via GDS
        gds_file = torch.cuda.gds.GdsFile(path, os.O_RDONLY)
        for name, tensor in loaded_dict.items():
            gds_file.load_storage(tensor.untyped_storage(), offset=offsets[name])
    end_load = time.time()

    return (end_save - start_save) / iterations, (end_load - start_load) / iterations

# ================= EXECUTION =================

if __name__ == "__main__":
    MODEL_SIZE_GB = 2.0  # Adjust as needed
    ITERATIONS = 5
    STD_PATH = "model_std.pt"
    GDS_PATH = "model_gds.pt"

    if not torch.cuda.is_available():
        print("CUDA not found. GDS requires a GPU node.")
        exit()

    # Create dummy data
    my_model = create_heavy_model(MODEL_SIZE_GB)

    # Run Standard
    std_s, std_l = benchmark_standard(my_model, STD_PATH, ITERATIONS)
    
    # Run GDS
    gds_s, gds_l = benchmark_gds(my_model, GDS_PATH, ITERATIONS)

    # Cleanup files
    for p in [STD_PATH, GDS_PATH]:
        if os.path.exists(p): os.remove(p)

    # Results
    print("\n" + "="*30)
    print(f"RESULTS ({MODEL_SIZE_GB} GB File)")
    print("="*30)
    print(f"Standard Save: {std_s:.4f}s | GDS Save: {gds_s:.4f}s")
    print(f"Standard Load: {std_l:.4f}s | GDS Load: {gds_l:.4f}s")
    print("-" * 30)
    print(f"Load Speedup: {std_l/gds_l:.2f}x")
    print(f"Save Speedup: {std_s/gds_s:.2f}x")