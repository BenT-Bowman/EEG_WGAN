import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from random import randint
from scipy.signal import butter, filtfilt

# Define delta bandpass filter (0.5–4 Hz)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut=30, highcut=100, fs=256):
    b, a = butter_bandpass(lowcut, highcut, fs, order=4)
    return filtfilt(b, a, data)

# Parse arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--data', '-d', type=str, required=True, help='Path to the data (.npy) files.')
parser.add_argument('--fs', type=int, default=250, help='Sampling frequency (Hz), default is 250.')
args = parser.parse_args()

# Load and reshape EEG
data_path = args.data
eeg = np.load(data_path)
eeg = eeg.reshape(eeg.shape[0], 19, -1)
print(f"Loaded EEG shape: {eeg.shape}")

# Figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Plot delta-filtered EEG
def plot_eeg():
    target = randint(0, eeg.shape[0] - 1)
    ax.clear()
    for channel_data in eeg[target]:
        filtered = apply_bandpass_filter(channel_data, lowcut=0.5, highcut=4, fs=args.fs)
        ax.plot(filtered, linewidth=0.5)
    ax.set_title('Delta Band EEG (0.5–4 Hz)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    fig.canvas.draw_idle()

# Button click
def on_button_click(event):
    plot_eeg()

# Initial plot
plot_eeg()

# Add regenerate button
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, 'Regenerate')
button.on_clicked(on_button_click)

plt.show()
