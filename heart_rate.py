
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

# Load ECG data
record = wfdb.rdrecord("data/100")
fs = record.fs
ecg = record.p_signal[:, 0]

# Select 10 seconds
duration = 10
n_samples = int(duration * fs)
ecg = ecg[:n_samples]
time = np.arange(len(ecg)) / fs

# Filter ECG
sos = butter(
    4, [0.5, 40],
    btype="bandpass",
    fs=fs,
    output="sos"
)
filtered_ecg = sosfiltfilt(sos, ecg)

# Detect positive R-peaks
minimum_distance = int(0.25 * fs)
threshold = np.mean(filtered_ecg) + 0.5 * np.std(filtered_ecg)

peaks, properties = find_peaks(
    filtered_ecg,
    distance=minimum_distance,
    height=threshold,
    prominence=0.3 * np.std(filtered_ecg)
)

# Calculate heart rate
rr_intervals = np.diff(peaks) / fs

if len(rr_intervals) > 0:
    bpm = 60 / np.mean(rr_intervals)
    print("Estimated heart rate:", round(bpm, 2), "BPM")
else:
    print("Not enough R-peaks detected to calculate heart rate.")

print("Detected R-peaks:", len(peaks))

# Plot ECG and detected peaks
plt.figure(figsize=(12, 5))
plt.plot(time, filtered_ecg, label="Filtered ECG")
plt.plot(time[peaks], filtered_ecg[peaks], "ro", label="Detected R-peaks")
plt.title("ECG R-Peak Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ecg_r_peaks.png", dpi=150)
plt.show()
