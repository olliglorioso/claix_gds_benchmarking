from tqdm import tqdm
import torch
import torch.nn as nn
import time
import os
import gc
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.serialization import skip_data

import torch
from torch.utils.serialization import config as serialization_config

serialization_config.save.storage_alignment = 4096
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
    print(model_tensors, "model tensors")
    return model_tensors

def get_checkpoint_offsets(path):

    """

    Uses FakeTensorMode to identify where each tensor's data

    is located within the checkpoint file.

    """

    with FakeTensorMode():

        fake_state_dict = torch.load(path, weights_only=True)

    offsets = {}
    for name, tensor in fake_state_dict.items():
        storage = tensor.untyped_storage()
        if hasattr(storage, "_checkpoint_offset"):
            offsets[name] = storage._checkpoint_offset
        else:
            print("NO OFFSET FOUND")


    return offsets
# ================= BENCHMARK FUNCTIONS =================

def benchmark_standard(state_dict, path, iterations):
    print(f"\n--- Benchmarking Standard PyTorch ({iterations} iterations) ---")
    
    # Save Benchmark
    start_save = time.time()
    for _ in tqdm(range(iterations), desc="Standard Save"):

        torch.save(state_dict, path)
    end_save = time.time()
    
    # Load Benchmark
    start_load = time.time()
    for _ in tqdm(range(iterations), desc="Standard Load"):
        _ = torch.load(path, map_location="cuda", weights_only=True)
    end_load = time.time()
    
    return (end_save - start_save) / iterations, (end_load - start_load) / iterations

def benchmark_gds(state_dict, path, iterations):
    print(f"\n--- Benchmarking GDS Prototype ({iterations} iterations) ---")
    
    # 1. PRE-STEP: Reserve space and get offsets (only needed once)
    with skip_data():
        torch.save(state_dict, path)
    offsets = get_checkpoint_offsets(path)
    
    # 2. GDS SAVE BENCHMARK
    for _ in tqdm(range(iterations), desc="GDS Save"):

        gds_file = torch.cuda.gds.GdsFile(path, os.O_WRONLY)

        for name, tensor in tqdm(
            state_dict.items(),
            desc="Writing tensors",
            leave=False,
        ):
            gds_file.save_storage(
                tensor.untyped_storage(),
                offset=offsets[name],
            )


    # 3. GDS LOAD BENCHMARK
    for _ in tqdm(range(iterations), desc="GDS Load"):

        with skip_data():
            loaded_dict = torch.load(path, weights_only=True)

        gds_file = torch.cuda.gds.GdsFile(path, os.O_RDONLY)

        for name, tensor in tqdm(
            loaded_dict.items(),
            desc="Loading tensors",
            leave=False,
        ):
            gds_file.load_storage(
                tensor.untyped_storage(),
                offset=offsets[name],
            )
    end_save = time.time()

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
