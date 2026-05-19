import numpy as np
import argparse 
from random import randint
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--data', '-d', type=str, required=True, help='Path to the data (.npy) files.')
args = parser.parse_args()
data_path = args.data

eeg = np.load(data_path)
eeg = eeg.reshape(eeg.shape[0], 19, -1)

print(eeg.shape)

# Function to plot EEG data
def plot_eeg():
    target = randint(0, eeg.shape[0] - 1)
    ax.clear()  # Clear the main plot axes
    for channel_data in eeg[target]:
        ax.plot(channel_data, linewidth=0.5)  # Plot each channel
    ax.set_title('EEG Data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    fig.canvas.draw_idle()  # Redraw the figure

# Button click event handler
def on_button_click(event):
    plot_eeg()

# Create the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Adjust space for the button
plot_eeg()

# Add a button
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # Position [left, bottom, width, height]
button = Button(ax_button, 'Regenerate')

# Connect the button click event to the handler
button.on_clicked(on_button_click)

plt.show()

eeg = eeg.reshape(eeg.shape[0], 19, -1)

from scipy.signal import spectrogram
fs = 256  # Sampling frequency

# Create subplots for 19 channels (e.g., a 5x4 grid)
fig, axes = plt.subplots(5, 4, figsize=(15, 12))
axes = axes.flatten()

for i in range(19):
    # Compute spectrogram for the i-th channel (1D signal)
    f, t, Sxx = spectrogram(eeg[randint(0, eeg.shape[0]), i, :], fs=fs, nperseg=128, noverlap=64)
    
    # Convert power to dB scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    
    ax = axes[i]
    # Now Sxx_dB has shape (frequencies, time) i.e., (65, 15)
    pcm = ax.pcolormesh(t, f, Sxx_dB, shading='auto', cmap='viridis')
    
    ax.set_title(f'Channel {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    # Optionally add a colorbar
    if i % 4 == 0:
        fig.colorbar(pcm, ax=ax, orientation='vertical')

# Remove any unused subplots (if any)
for j in range(19, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()