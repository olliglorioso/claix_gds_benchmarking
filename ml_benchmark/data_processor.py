import os
import shutil
from tqdm import tqdm

# --- Configuration ---
MULTIPLIER = 100  # How many times to copy the dataset

# Source: The original DALI_extra MRI slices
SOURCE_DIR = os.path.join(
    os.environ['DALI_EXTRA_PATH'], 
    'db', '3D', 'MRI', 'Knee', 'npy_2d_slices', 'STU00001'
)

# Destination: Your high-speed NVMe workspace on CLAIX
# Fallback to current directory if BEEOND isn't active
DEST_DIR = os.path.join(
    os.environ.get("BEEOND", os.environ.get("HPCWORK", ".")), 
    "massive_gds_dataset"
)

def multiply_dataset():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    original_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.npy')])
    num_originals = len(original_files)
    total_new_files = num_originals * MULTIPLIER

    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Original files: {num_originals}")
    print(f"Generating {total_new_files} total files...\n")

    # Copy files
    with tqdm(total=total_new_files, desc="Copying dataset") as pbar:
        for i in range(MULTIPLIER):
            for orig_file in original_files:
                src_path = os.path.join(SOURCE_DIR, orig_file)
                
                # Create a new unique filename (e.g., 0001_slice_0.npy)
                new_filename = f"{i:04d}_{orig_file}"
                dst_path = os.path.join(DEST_DIR, new_filename)
                
                shutil.copy2(src_path, dst_path)
                pbar.update(1)

    print(f"\nDone! Your massive dataset is ready at:")
    print(DEST_DIR)

if __name__ == "__main__":
    multiply_dataset()