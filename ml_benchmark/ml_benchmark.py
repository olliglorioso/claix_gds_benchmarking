from tqdm import tqdm
import torch
import time
import os
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.serialization import skip_data
from torch.utils.serialization import config as serialization_config

MODEL_SIZE_GB = 10.0
ITERATIONS = 10
BEEOND_DIR = os.getenv("BEEOND", ".")

STD_PATH = os.path.join(BEEOND_DIR, "model_std.pt")
GDS_PATH = os.path.join(BEEOND_DIR, "model_gds.pt")
serialization_config.save.storage_alignment = 4096


def create_heavy_model(size_gb):
    """
    Creates a large GPU state_dict approximately equal to size_gb.
    """
    total_params = int((size_gb * (1024 ** 3)) / 4)
    print(f"Creating model with {total_params:,} parameters (~{size_gb} GB)...")

    model_tensors = {
        f"layer_{i}": torch.randn(
            total_params // 10,
            device="cuda",
            dtype=torch.float32,
        )
        for i in range(10)
    }
    return model_tensors

def get_checkpoint_offsets(path):
    """
    Uses FakeTensorMode to get storage offsets
    inside the checkpoint file.
    """
    with FakeTensorMode():
        fake_state_dict = torch.load(
            path,
            weights_only=True,
        )

    offsets = {}

    for name, tensor in fake_state_dict.items():
        storage = tensor.untyped_storage()
        if not hasattr(storage, "_checkpoint_offset"):
            raise RuntimeError(
                f"No checkpoint offset found for tensor: {name}"
            )
        offsets[name] = storage._checkpoint_offset
    return offsets

def benchmark_standard(state_dict, path, iterations):

    print(f"\n--- Benchmarking Standard PyTorch ({iterations} iterations) ---")
    save_times = []
    for _ in tqdm(range(iterations), desc="Standard Save"):
        torch.cuda.synchronize()
        start = time.time()
        torch.save(state_dict, path)
        torch.cuda.synchronize()
        end = time.time()
        save_times.append(end - start)

    load_times = []
    for _ in tqdm(range(iterations), desc="Standard Load"):
        torch.cuda.synchronize()
        start = time.time()
        loaded = torch.load(
            path,
            map_location="cuda",
            weights_only=True,
        )
        torch.cuda.synchronize()
        end = time.time()
        load_times.append(end - start)
        del loaded
    avg_save = sum(save_times) / len(save_times)
    avg_load = sum(load_times) / len(load_times)
    return avg_save, avg_load

def benchmark_gds(state_dict, path, iterations):
    print(f"\n--- Benchmarking GDS Prototype ({iterations} iterations) ---")
    print("Preparing checkpoint skeleton...")
    with skip_data():
        torch.save(state_dict, path)

    offsets = get_checkpoint_offsets(path)
    save_times = []
    gds_file = torch.cuda.gds.GdsFile(path, os.O_RDWR)

    try:
        for _ in tqdm(range(iterations), desc="GDS Save"):
            torch.cuda.synchronize()
            start = time.time()
            for name, tensor in state_dict.items():
                gds_file.save_storage(
                    tensor.untyped_storage(),
                    offsets[name],
                )
            torch.cuda.synchronize()
            end = time.time()
            save_times.append(end - start)

    finally:
        del gds_file

    load_times = []
    gds_file = torch.cuda.gds.GdsFile(path, os.O_RDONLY)
    try:
        for _ in tqdm(range(iterations), desc="GDS Load"):
            # Create empty tensors only
            with skip_data():
                loaded_dict = torch.load(
                    path,
                    weights_only=True,
                )
            torch.cuda.synchronize()
            start = time.time()
            for name, tensor in loaded_dict.items():
                gds_file.load_storage(
                    tensor.untyped_storage(),
                    offsets[name],
                )
            torch.cuda.synchronize()
            end = time.time()
            load_times.append(end - start)
            for k in state_dict:
                if not torch.equal(
                    state_dict[k],
                    loaded_dict[k],
                ):
                    raise RuntimeError(
                        f"Mismatch detected in tensor: {k}"
                    )
            del loaded_dict
    finally:
        del gds_file

    avg_save = sum(save_times) / len(save_times)
    avg_load = sum(load_times) / len(load_times)

    return avg_save, avg_load


# ============================================================
# RESULTS
# ============================================================

def print_results(
    model_size_gb,
    std_save,
    std_load,
    gds_save,
    gds_load,
):

    print("\n" + "=" * 50)
    print(f"RESULTS ({model_size_gb} GB)")
    print("=" * 50)

    print(f"Standard Save : {std_save:.4f} s")
    print(f"GDS Save      : {gds_save:.4f} s")

    print()

    print(f"Standard Load : {std_load:.4f} s")
    print(f"GDS Load      : {gds_load:.4f} s")

    print("-" * 50)

    print(f"Save Speedup  : {std_save / gds_save:.2f}x")
    print(f"Load Speedup  : {std_load / gds_load:.2f}x")

    print("-" * 50)

    std_save_bw = model_size_gb / std_save
    gds_save_bw = model_size_gb / gds_save

    std_load_bw = model_size_gb / std_load
    gds_load_bw = model_size_gb / gds_load

    print(f"Standard Save BW : {std_save_bw:.2f} GB/s")
    print(f"GDS Save BW      : {gds_save_bw:.2f} GB/s")

    print()

    print(f"Standard Load BW : {std_load_bw:.2f} GB/s")
    print(f"GDS Load BW      : {gds_load_bw:.2f} GB/s")

    print("=" * 50)

if __name__ == "__main__":
    import csv
    import gc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    if not hasattr(torch.cuda, "gds"):
        raise RuntimeError(
            "torch.cuda.gds not available.\n"
            "Requires PyTorch >= 2.7 with GDS support."
        )

    TEST_SIZES = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0] 
    CSV_FILENAME = "gds_benchmark_sweep.csv"

    # Initialize the CSV file and write the header
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model_Size_GB", 
            "Std_Save_s", "Std_Load_s", 
            "GDS_Save_s", "GDS_Load_s",
            "Std_Save_GBs", "Std_Load_GBs",
            "GDS_Save_GBs", "GDS_Load_GBs"
        ])

    print(f"Starting sweep across sizes: {TEST_SIZES}")

    for size in TEST_SIZES:
        print(f"\n{'#'*60}")
        print(f"### TESTING SIZE: {size} GB")
        print(f"{'#'*60}")

        # 1. Create a fresh model for this size
        my_model = create_heavy_model(size)
        torch.cuda.synchronize()

        # 2. Execute benchmarks using your defined functions
        # This will run for the number of ITERATIONS defined globally
        std_save_avg, std_load_avg = benchmark_standard(
            my_model, 
            STD_PATH, 
            ITERATIONS
        )

        gds_save_avg, gds_load_avg = benchmark_gds(
            my_model, 
            GDS_PATH, 
            ITERATIONS
        )

        # 3. Print results to the console for real-time monitoring
        print_results(
            size,
            std_save_avg,
            std_load_avg,
            gds_save_avg,
            gds_load_avg,
        )

        # 4. Append the results to the CSV (assuming CSV was intended over CSS)
        with open(CSV_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                size, 
                std_save_avg, std_load_avg, 
                gds_save_avg, gds_load_avg,
                size / std_save_avg, size / std_load_avg,
                size / gds_save_avg, size / gds_load_avg
            ])

        # 5. Crucial: Clear VRAM and local storage before the next larger iteration
        del my_model
        torch.cuda.empty_cache()
        gc.collect()

        for p in [STD_PATH, GDS_PATH]:
            if os.path.exists(p):
                os.remove(p)

    print(f"\nSweep complete. All results saved to {CSV_FILENAME}")