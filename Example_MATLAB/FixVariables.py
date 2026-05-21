import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import medfilt

# === USER INPUTS ===
input_folder = r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data_raw"
output_folder = r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS"

# Conversion factors
pixel_to_meter = 3.2e-6       # meters per pixel
fps = 10_000_000              # frames per second

# Spike removal: a frame is removed if its value exceeds this multiple of the
# local median (computed with a window-3 median filter over the full array).
SPIKE_THRESHOLD = 2.0


def remove_spikes(R):
    """Remove isolated spike frames using median-filter comparison.

    A frame is a spike if R[i] > SPIKE_THRESHOLD * median(R[i-1], R[i], R[i+1]).
    """
    if len(R) < 3:
        return R, 0
    R_med = medfilt(R.astype(float), kernel_size=3)
    # Where median is near zero, ratio is unreliable — treat as non-spike
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(R_med > 0, R / R_med, 1.0)
    mask = ratio <= SPIKE_THRESHOLD
    return R[mask], int(np.sum(~mask))


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all .mat files
for filename in os.listdir(input_folder):
    if filename.endswith(".mat"):
        file_path = os.path.join(input_folder, filename)

        # Load .mat file
        data = loadmat(file_path)

        # Check if 'Radius' exists
        if 'Radius' not in data:
            print(f"Skipping {filename}: 'Radius' not found")
            continue

        Radius = data['Radius'].squeeze()  # remove extra dimensions if needed

        # === Remove spike frames ===
        Radius, n_removed = remove_spikes(Radius)
        if n_removed > 0:
            print(f"  Removed {n_removed} spike frame(s) from {filename}")

        # === Conversions ===
        R = Radius * pixel_to_meter
        t = np.arange(len(Radius)) / fps

        # === Remove negative radii ===
        nonnegative_mask = R >= 0
        n_negative = int(np.sum(~nonnegative_mask))
        if n_negative > 0:
            R = R[nonnegative_mask]
            t = t[nonnegative_mask]
            print(f"  Removed {n_negative} negative R frame(s) from {filename}")

        # Save new variables
        output_data = {
            'R': R,
            't': t
        }

        output_filename = os.path.splitext(filename)[0] + "_converted.mat"
        output_path = os.path.join(output_folder, output_filename)

        savemat(output_path, output_data)

        print(f"Processed: {filename} → {output_filename}")

print("All files processed.")
