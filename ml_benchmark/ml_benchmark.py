from tqdm import tqdm
import torch
import time
import os
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.serialization import skip_data
from torch.utils.serialization import config as serialization_config

# ============================================================
# CONFIG
# ============================================================

MODEL_SIZE_GB = 2.0
ITERATIONS = 5

STD_PATH = "model_std.pt"
GDS_PATH = "model_gds.pt"

# Required for GPUDirect Storage
serialization_config.save.storage_alignment = 4096

# ============================================================
# MODEL CREATION
# ============================================================

def create_heavy_model(size_gb):
    """
    Creates a large GPU state_dict approximately equal to size_gb.
    """

    # FP32 = 4 bytes
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


# ============================================================
# CHECKPOINT OFFSETS
# ============================================================

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


# ============================================================
# STANDARD PYTORCH BENCHMARK
# ============================================================

def benchmark_standard(state_dict, path, iterations):

    print(f"\n--- Benchmarking Standard PyTorch ({iterations} iterations) ---")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_times = []

    for _ in tqdm(range(iterations), desc="Standard Save"):

        torch.cuda.synchronize()

        start = time.time()

        torch.save(state_dict, path)

        torch.cuda.synchronize()

        end = time.time()

        save_times.append(end - start)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

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


# ============================================================
# GPUDIRECT STORAGE BENCHMARK
# ============================================================

def benchmark_gds(state_dict, path, iterations):

    print(f"\n--- Benchmarking GDS Prototype ({iterations} iterations) ---")

    # --------------------------------------------------------
    # CREATE CHECKPOINT SKELETON
    # --------------------------------------------------------

    print("Preparing checkpoint skeleton...")

    with skip_data():
        torch.save(state_dict, path)

    offsets = get_checkpoint_offsets(path)

    # --------------------------------------------------------
    # GDS SAVE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GDS LOAD
    # --------------------------------------------------------

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

            # Correctness validation
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


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    if not hasattr(torch.cuda, "gds"):
        raise RuntimeError(
            "torch.cuda.gds not available.\n"
            "Requires PyTorch >= 2.7 with GDS support."
        )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    my_model = create_heavy_model(MODEL_SIZE_GB)

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # STANDARD BENCHMARK
    # --------------------------------------------------------

    std_s, std_l = benchmark_standard(
        my_model,
        STD_PATH,
        ITERATIONS,
    )

    # --------------------------------------------------------
    # GDS BENCHMARK
    # --------------------------------------------------------

    gds_s, gds_l = benchmark_gds(
        my_model,
        GDS_PATH,
        ITERATIONS,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print_results(
        MODEL_SIZE_GB,
        std_s,
        std_l,
        gds_s,
        gds_l,
    )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    for p in [STD_PATH, GDS_PATH]:

        if os.path.exists(p):
            os.remove(p)

    print("\nDone.")
