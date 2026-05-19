import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

real_path = r"training_data\5_sec_seq_1_sec_skip\Control\H S1 EC.npy"
synth_path = r"gen_20\control\generated_data_0.npy"

real = np.load(real_path)        # shape: (channels, time)
synth = np.load(synth_path)      # shape: (channels, time)

x_tensor = torch.from_numpy(synth)          # (num_samples_to_generate, 9500)
x_restored = x_tensor.view(-1, 19, 1025)
synth = x_restored.numpy()

real = real[0]
synth = synth[0]

n_channels, T = real.shape

# Use only first half of the time series
half = T // 2
real = real[:, :half]
synth = synth[:, :half]

# Ensure shapes are correct
assert real.ndim == 2, f"Real EEG expected shape (C,T), got {real.shape}"
assert synth.ndim == 2, f"Synth EEG expected shape (C,T), got {synth.shape}"

n_channels, T = real.shape

cmap = cmap = mpl.cm.get_cmap("plasma")   # smooth gradient
colors = [cmap(i / n_channels) for i in range(n_channels)]

def plot_eeg(data, title):
    n_channels, T = data.shape
    fig, ax = plt.subplots(figsize=(14, 10))

    offset = 5 * np.std(data)  # good spacing based on signal magnitude

    for ch in range(n_channels):
        ax.plot(
            data[ch] + ch * offset,
            color=colors[ch],
            linewidth=1.2
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Channels (offset)")
    ax.set_yticks([])  # cleaner
    ax.grid(False)

    plt.tight_layout()
    plt.show()

plot_eeg(real, "")
plot_eeg(synth, "")
