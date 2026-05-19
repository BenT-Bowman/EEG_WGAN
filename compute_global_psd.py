import numpy as np
import glob
from scipy.signal import welch
import torch

# ---------------------------------------------------------
# USER SETS THESE TWO LINES ONLY
# ---------------------------------------------------------
real_files  = glob.glob(r"training_data\5_sec_seq_1_sec_skip\Patient\*.npy")
synth_files = glob.glob(r"gen_20\patient\generated_data_*.npy")
# ---------------------------------------------------------

FS = 256
NPERSEG = 256
NOVERLAP = 128
FREQ_MIN = 1
FREQ_MAX = 40

def reshape_to_19_1025(sample):
    """
    Ensure the window is shaped (19, 1025) using the PyTorch trick.
    """
    x_tensor = torch.from_numpy(sample)
    x_restored = x_tensor.view(19, 1025)
    return x_restored.numpy()

def compute_psd_set(file_list):
    """
    Loads all .npy files, reshapes each window to (19,1025) if needed,
    computes PSD for every window & channel.
    Returns freqs, psd_mean, psd_std.
    """
    psd_collection = []

    for path in file_list:
        data = np.load(path)   # shape: (num_windows, *)
        
        for window in data:    # shape: (*,)
            if window.shape != (19, 1025):
                window = reshape_to_19_1025(window)
            for ch in range(window.shape[0]):
                freqs, psd = welch(
                    window[ch],
                    fs=FS,
                    nperseg=NPERSEG,
                    noverlap=NOVERLAP
                )
                psd_collection.append(psd)

    psd_collection = np.array(psd_collection)   # (N_total, freq_bins)

    # Restrict to 1–40 Hz
    idx = np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]
    freqs = freqs[idx]
    psd_collection = psd_collection[:, idx]

    psd_mean = psd_collection.mean(axis=0)
    psd_std  = psd_collection.std(axis=0)

    return freqs, psd_mean, psd_std


# ---- Compute PSD summaries ----
freqs, psd_real_mean,  psd_real_std  = compute_psd_set(real_files)
_,     psd_syn_mean,   psd_syn_std   = compute_psd_set(synth_files)

# ---- Save to a single NPZ ----
np.savez(
    "mdd_psd_summary_fig4.npz",
    freqs=freqs,
    psd_real_mean=psd_real_mean,
    psd_real_std=psd_real_std,
    psd_syn_mean=psd_syn_mean,
    psd_syn_std=psd_syn_std,
)

print("Saved PSD summary → psd_summary_fig4.npz")
