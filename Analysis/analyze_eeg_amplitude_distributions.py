import numpy as np
import torch
import glob
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt



real_files  = glob.glob(r"")
synth_files = glob.glob(r"")
save_path   = "amp_distributions_control.npz"



def load_eeg_data(file_list):
    """
    Loads all .npy files and reshapes each sample to (19, 1025).
    Returns: (num_windows, 19, 1025)
    """
    all_data = []

    for path in file_list:
        data = np.load(path)
        for sample in data:
            if sample.shape != (19, 1025):
                sample = torch.from_numpy(sample).view(19, 1025).numpy()
            all_data.append(sample)

    return np.array(all_data)


real_data  = load_eeg_data(real_files)
synth_data = load_eeg_data(synth_files)


all_values = np.concatenate([real_data.reshape(-1),
                             synth_data.reshape(-1)])

num_bins = 100
amp_bins = np.linspace(all_values.min(), all_values.max(), num_bins)


def compute_histograms(data, bins):
    """
    Returns (n_channels, n_bins-1) normalized histograms.
    """
    n_channels = data.shape[1]
    out = np.zeros((n_channels, len(bins) - 1))

    for ch in range(n_channels):
        flat = data[:, ch, :].reshape(-1)
        hist, _ = np.histogram(flat, bins=bins, density=True)
        out[ch] = hist

    return out


hist_real  = compute_histograms(real_data, amp_bins)
hist_synth = compute_histograms(synth_data, amp_bins)


# Per channel js divergences
def js_divergences(real_h, synth_h):
    js = []
    for ch in range(real_h.shape[0]):
        p = real_h[ch] + 1e-12
        q = synth_h[ch] + 1e-12
        js.append(jensenshannon(p, q, base=2.0))
    return np.array(js)


js_amp = js_divergences(hist_real, hist_synth)


np.savez(
    save_path,
    amp_bins=amp_bins,
    hist_real=hist_real,
    hist_synth=hist_synth,
    js_amp=js_amp,
)

print(f"Saved amplitude distributions → {save_path}")


plt.figure(figsize=(10, 6))

mean_real  = hist_real.mean(axis=0)
mean_synth = hist_synth.mean(axis=0)

plt.plot(amp_bins[:-1], mean_real,  label="Real (mean over channels)", linewidth=2)
plt.plot(amp_bins[:-1], mean_synth, label="Synthetic (mean over channels)", linewidth=2, linestyle="--")

plt.xlabel("Amplitude")
plt.ylabel("Density")
plt.title("Amplitude Distribution (Real vs Synthetic)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
