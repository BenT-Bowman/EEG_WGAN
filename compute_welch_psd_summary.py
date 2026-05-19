import numpy as np
from scipy.signal import welch

data_real_ctrl = {}
data_synth_ctrl = {}
data_real_mdd  = {}
data_synth_mdd = {}

# Settings

fs = 256
nperseg = 256
noverlap = 128
fmin, fmax = 1, 40

def compute_fold_psd(data_windows):
    """
    data_windows: (num_windows, n_channels, n_samples)
    Returns: one PSD curve averaged across windows & channels.
    Compute PSD averaged across windows & channels
    """
    psds = []

    for win in data_windows:  # shape: (19, samples)
        for ch in win:        # loop channels
            freqs, psd = welch(
                ch,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap
            )
            psds.append(psd)

    psds = np.array(psds)  # (num_windows*19, freqs)
    mean_psd = psds.mean(axis=0)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[freq_mask], mean_psd[freq_mask]


# Process all folds

folds = sorted(data_real_ctrl.keys())

psd_real_ctrl_fold = []
psd_synth_ctrl_fold = []
psd_real_mdd_fold = []
psd_synth_mdd_fold = []

for f in folds:
    freqs, p = compute_fold_psd(data_real_ctrl[f])
    psd_real_ctrl_fold.append(p)

    _, p = compute_fold_psd(data_synth_ctrl[f])
    psd_synth_ctrl_fold.append(p)

    _, p = compute_fold_psd(data_real_mdd[f])
    psd_real_mdd_fold.append(p)

    _, p = compute_fold_psd(data_synth_mdd[f])
    psd_synth_mdd_fold.append(p)

psd_real_ctrl_fold = np.array(psd_real_ctrl_fold)
psd_synth_ctrl_fold = np.array(psd_synth_ctrl_fold)
psd_real_mdd_fold  = np.array(psd_real_mdd_fold)
psd_synth_mdd_fold = np.array(psd_synth_mdd_fold)


# Average Across folds

psd_real_ctrl_mean = psd_real_ctrl_fold.mean(axis=0)
psd_real_ctrl_std  = psd_real_ctrl_fold.std(axis=0)

psd_synth_ctrl_mean = psd_synth_ctrl_fold.mean(axis=0)
psd_synth_ctrl_std  = psd_synth_ctrl_fold.std(axis=0)

psd_real_mdd_mean = psd_real_mdd_fold.mean(axis=0)
psd_real_mdd_std  = psd_real_mdd_fold.std(axis=0)

psd_synth_mdd_mean = psd_synth_mdd_fold.mean(axis=0)
psd_synth_mdd_std  = psd_synth_mdd_fold.std(axis=0)


np.savez(
    "psd_summary.npz",
    freqs=freqs,
    psd_real_ctrl_mean=psd_real_ctrl_mean,
    psd_real_ctrl_std=psd_real_ctrl_std,
    psd_synth_ctrl_mean=psd_synth_ctrl_mean,
    psd_synth_ctrl_std=psd_synth_ctrl_std,
    psd_real_mdd_mean=psd_real_mdd_mean,
    psd_real_mdd_std=psd_real_mdd_std,
    psd_synth_mdd_mean=psd_synth_mdd_mean,
    psd_synth_mdd_std=psd_synth_mdd_std,
)

print("Saved PSD summary → psd_summary.npz")
