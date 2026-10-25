
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# Step 1: Load the ECG recording
record = wfdb.rdrecord("data/100")

fs = record.fs
ecg = record.p_signal[:, 0]

print("ECG loaded successfully!")
print("Sampling frequency:", fs, "Hz")

# Step 2: Select the first 10 seconds
duration = 10
n_samples = int(duration * fs)

ecg_segment = ecg[:n_samples]
time = np.arange(n_samples) / fs

# Step 3: Design a bandpass filter
sos = butter(
    4,
    [0.5, 40],
    btype="bandpass",
    fs=fs,
    output="sos"
)

# Step 4: Filter the ECG signal
filtered_ecg = sosfiltfilt(sos, ecg_segment)

print("ECG filtering completed!")

# Step 5: Plot original and filtered signals
plt.figure(figsize=(12, 7))

plt.subplot(2, 1, 1)
plt.plot(time, ecg_segment)
plt.title("Original ECG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, filtered_ecg)
plt.title("Filtered ECG Signal (0.5–40 Hz)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)

plt.tight_layout()

# Step 6: Save the graph
plt.savefig("ecg_signal.png", dpi=150)

print("Graph saved as ecg_signal.png")

plt.show()
