import numpy as np
import torch
import glob
from scipy.signal import correlate, hilbert
from scipy.stats import pearsonr
import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr

real_files  = glob.glob(r"training_data\5_sec_seq_1_sec_skip\Patient\*.npy")
synth_files = glob.glob(r"gen_20\patient\generated_data_*.npy")

def load_eeg_data(file_list):
    """
    Load all .npy files and reshape samples to (19,1025) if needed.
    Returns concatenated array: (num_windows_total, 19, 1025)
    """
    all_data = []

    for path in file_list:
        data = np.load(path)  # shape: (num_windows, ?)
        for sample in data:
            if sample.shape != (19, 1025):
                sample = torch.from_numpy(sample).view(19, 1025).numpy()
            all_data.append(sample)

    return np.array(all_data)

real_data  = load_eeg_data(real_files)
synth_data = load_eeg_data(synth_files)
sfreq = 256

import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr

def compute_lag_envelopes_distribution(real, synth, sfreq, mode='full'):
    """
    Compute cross-correlation envelopes for two EEG datasets as distributions.
    
    Parameters
    ----------
    real : np.ndarray
        Real EEG, shape (n_real_samples, n_channels, n_times)
    synth : np.ndarray
        Synthetic EEG, shape (n_synth_samples, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    mode : str
        'full', 'valid', or 'same' for correlation
        
    Returns
    -------
    lag_times : np.ndarray
        Lag axis in seconds
    mean_env_real : np.ndarray
        Mean envelope over real samples and channels
    std_env_real : np.ndarray
        Std envelope over real samples and channels
    mean_env_synth : np.ndarray
        Mean envelope over synthetic samples and channels
    std_env_synth : np.ndarray
        Std envelope over synthetic samples and channels
    rho : float
        Pearson correlation between mean envelopes (real vs synth)
    """
    n_real, n_channels, n_times_real = real.shape
    n_synth, _, n_times_synth = synth.shape

    # Function to compute per-sample, per-channel envelopes
    def envelopes_array(data):
        envelopes = []
        for sample in data:
            sample_env = []
            for ch in range(sample.shape[0]):
                x = sample[ch] - sample[ch].mean()
                cc = correlate(x, x, mode=mode)
                sample_env.append(np.abs(cc))
            envelopes.append(sample_env)
        return np.array(envelopes)  # shape: (n_samples, n_channels, n_lags)

    # Compute envelopes
    env_real = envelopes_array(real)
    env_synth = envelopes_array(synth)

    # Lag axis (same length for all)
    n_lags = env_real.shape[2]
    lags = correlation_lags(n_times_real, n_times_real, mode=mode)
    lag_times = lags / sfreq

    # Mean/std across samples and channels
    mean_env_real = env_real.mean(axis=(0,1))
    std_env_real = env_real.std(axis=(0,1))
    mean_env_synth = env_synth.mean(axis=(0,1))
    std_env_synth = env_synth.std(axis=(0,1))

    # Pearson correlation between mean envelopes
    rho, _ = pearsonr(mean_env_real, mean_env_synth)

    return lag_times, mean_env_real, std_env_real, mean_env_synth, std_env_synth, rho

lag_times, mean_env_real, std_env_real, mean_env_synth, std_env_synth, rho = compute_lag_envelopes_distribution(real_data, synth_data, sfreq=256)

# Save to .npz
np.savez('lag_envelopes_real_vs_synth.npz',
         lag_times=lag_times,
         mean_env_real=mean_env_real,
         std_env_real=std_env_real,
         mean_env_synth=mean_env_synth,
         std_env_synth=std_env_synth,
         rho=rho)
