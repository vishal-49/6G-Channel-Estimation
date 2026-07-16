import h5py
import numpy as np
import os

# ===========================
# Configuration
# ===========================

DATASET_PATH = "6G_ChanEst_Dataset_10k_Samples.h5"   # Change if required
OUTPUT_DIR = "samples"

NUM_SAMPLES = 100      # Number of samples to export

# ===========================
# Create output folder
# ===========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# Read dataset
# ===========================

with h5py.File(DATASET_PATH, "r") as f:

    X = f["X_input"]

    total = min(NUM_SAMPLES, len(X))

    for i in range(total):
        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"X_input_sample_{i}.npy"
            ),
            X[i]
        )

print(f"\n✅ Successfully exported {total} samples.")
