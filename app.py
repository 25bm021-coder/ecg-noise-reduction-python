
import streamlit as st
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ECG AI Analyzer",
    page_icon="❤️",
    layout="wide"
)

st.title("ECG AI-Based Abnormality Analysis")
st.caption(
    "Biomedical Engineering Project | "
    "ECG Signal Processing and AI Research Prototype"
)

st.warning(
    "Research prototype only. ECG measurements and AI predictions "
    "are not medical diagnoses."
)

DATA_FOLDER = Path("data")
MODEL_PATH = Path("models/ecg_rf_model.joblib")

# =====================================================
# 2. LOAD MODEL
# =====================================================

model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        st.sidebar.error(f"Model loading failed: {error}")

# =====================================================
# 3. SIDEBAR
# =====================================================

st.sidebar.header("Analysis Settings")

analysis_mode = st.sidebar.radio(
    "Select ECG source",
    [
        "Upload ECG CSV",
        "Analyze MIT-BIH Recording"
    ]
)

# =====================================================
# 4. FILTER ECG AND DETECT R-PEAKS
# =====================================================

def process_ecg(ecg, fs):

    ecg = np.asarray(ecg, dtype=float).flatten()

    if not np.all(np.isfinite(ecg)):
        raise ValueError(
            "ECG contains missing or invalid values. "
            "Clean the signal before analysis."
        )

    if fs <= 1:
        raise ValueError("Invalid sampling frequency.")

    if len(ecg) < max(10, int(2 * fs)):
        raise ValueError(
            "Please provide at least 2 seconds of ECG samples."
        )

    if np.std(ecg) < 1e-10:
        raise ValueError("The ECG signal is flat.")

    # Butterworth bandpass filter: 0.5 to 40 Hz
    low_cutoff = 0.5
    high_cutoff = min(40.0, fs * 0.45)

    if high_cutoff <= low_cutoff:
        raise ValueError(
            "Sampling frequency is too low for this filter."
        )

    sos = butter(
        4,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    # Apply filter without overwriting the original signal
    filtered = sosfiltfilt(sos, ecg)

    # Basic positive R-peak detection
    minimum_distance = max(1, int(0.25 * fs))
    prominence = max(0.3 * np.std(filtered), 1e-8)

    peaks, _ = find_peaks(
        filtered,
        distance=minimum_distance,
        prominence=prominence
    )

    # Preliminary heart-rate estimate
    bpm = None

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs

        valid_rr = rr_intervals[
            (rr_intervals >= 60 / 220) &
            (rr_intervals <= 60 / 35)
        ]

        if len(valid_rr) > 0:
            bpm = 60.0 / np.median(valid_rr)

    return filtered, peaks, bpm


# =====================================================
# 5. DISPLAY ORIGINAL VS FILTERED ECG
# =====================================================

def display_analysis(ecg, fs, source_name):

    try:
        # Keep the original signal for comparison
        original = np.asarray(ecg, dtype=float).flatten()

        filtered, peaks, bpm = process_ecg(original, fs)

        time = np.arange(len(original)) / fs

        st.subheader("ECG Signal Comparison")

        st.write(
            f"**Source:** {source_name}  \n"
            f"**Sampling frequency:** {fs:g} Hz  \n"
            f"**Duration:** {len(original) / fs:.2f} seconds"
        )

        # Two separate graphs
        fig, axes = plt.subplots(
            2, 1,
            figsize=(13, 8),
            sharex=True
        )

        # Graph 1: Original ECG
        axes[0].plot(
            time,
            original,
            color="royalblue",
            linewidth=0.8,
            label="Original (Unfiltered) ECG"
        )

        axes[0].set_title(
            "Original ECG Signal — Before Filtering"
        )
        axes[0].set_ylabel("Amplitude (input units)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        # Graph 2: Filtered ECG
        axes[1].plot(
            time,
            filtered,
            color="darkorange",
            linewidth=0.9,
            label="Filtered ECG (0.5–40 Hz)"
        )

        axes[1].plot(
            time[peaks],
            filtered[peaks],
            "ro",
            markersize=4,
            label="Detected R-peaks"
        )

        axes[1].set_title(
            "Filtered ECG Signal — After Filtering"
        )
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude (input units)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sampling Frequency",
            f"{fs:g} Hz"
        )

        col2.metric(
            "Estimated Heart Rate",
            f"{bpm:.1f} BPM" if bpm is not None else "Unavailable"
        )

        col3.metric(
            "Detected Peaks",
            str(len(peaks))
        )

        st.info(
            "The upper graph shows the original signal. "
            "The lower graph shows the signal after a 0.5–40 Hz "
            "Butterworth bandpass filter. Red dots indicate "
            "candidate peaks, which may include detection errors."
        )

        # Export both signals
        output_df = pd.DataFrame({
            "time_seconds": time,
            "original_ecg": original,
            "filtered_ecg": filtered,
            "detected_r_peak": np.isin(
                np.arange(len(filtered)), peaks
            )
        })

        st.download_button(
            "Download Original and Filtered ECG CSV",
            data=output_df.to_csv(index=False),
            file_name="ecg_comparison.csv",
            mime="text/csv"
        )

        return filtered, peaks

    except Exception as error:
        st.error(f"ECG analysis failed: {error}")
        return None, None


# =====================================================
# 6. CSV UPLOAD MODE
# =====================================================

if analysis_mode == "Upload ECG CSV":

    st.header("Upload Your ECG Recording")

    st.write(
        "Upload a CSV containing one ECG sample per row. "
        "You can use this feature later with recordings from "
        "your ECG sensor and ESP32."
    )

    uploaded_file = st.file_uploader(
        "Choose an ECG CSV file",
        type=["csv"]
    )

    sampling_rate = st.number_input(
        "Actual recording sampling frequency (Hz)",
        min_value=10,
        max_value=2000,
        value=360,
        step=10
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            numeric_columns = list(
                df.select_dtypes(include="number").columns
            )

            if not numeric_columns:
                st.error(
                    "No numeric columns found in the CSV."
                )
                st.stop()

            st.subheader("Uploaded Data Preview")
            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            ecg_column = st.selectbox(
                "Select the ECG amplitude column",
                options=numeric_columns,
                index=(
                    numeric_columns.index("ecg")
                    if "ecg" in numeric_columns
                    else 0
                )
            )

            if st.button(
                "Analyze Uploaded ECG",
                type="primary"
            ):

                ecg_values = pd.to_numeric(
                    df[ecg_column],
                    errors="coerce"
                ).to_numpy(dtype=float)

                # Remove invalid samples
                ecg_values = ecg_values[
                    np.isfinite(ecg_values)
                ]

                if len(ecg_values) < 2 * sampling_rate:
                    st.error(
                        "The CSV must contain at least "
                        "2 seconds of valid ECG samples."
                    )
                else:
                    filtered, peaks = display_analysis(
                        ecg_values,
                        float(sampling_rate),
                        uploaded_file.name
                    )

                    if filtered is not None:
                        st.subheader("AI Classification")

                        st.warning(
                            "AI classification is not enabled for "
                            "uploaded recordings. The current model "
                            "was trained using MIT-BIH heartbeat "
                            "segments and has not been validated for "
                            "your hardware signal."
                        )

        except Exception as error:
            st.error(f"Could not read the CSV: {error}")


# =====================================================
# 7. MIT-BIH DATASET MODE
# =====================================================

else:

    st.header("Analyze MIT-BIH ECG Recording")

    record_header = DATA_FOLDER / "100.hea"
    record_data = DATA_FOLDER / "100.dat"

    if not record_header.exists() or not record_data.exists():
        st.error(
            "Record 100 is missing. Check data/100.hea "
            "and data/100.dat."
        )
        st.stop()

    if st.button(
        "Analyze MIT-BIH Recording",
        type="primary"
    ):

        try:
            record = wfdb.rdrecord(
                str(DATA_FOLDER / "100")
            )

            ecg = record.p_signal[:, 0]
            fs = float(record.fs)

            # Show the first 10 seconds
            duration_samples = min(
                len(ecg),
                int(10 * fs)
            )

            filtered, peaks = display_analysis(
                ecg[:duration_samples],
                fs,
                "MIT-BIH Record 100"
            )

            if filtered is not None:

                st.subheader("AI Heartbeat Classification")

                if model is None:
                    st.warning(
                        "The saved AI model was not found. "
                        "Run train_model.py to train it first."
                    )
                else:
                    annotations = wfdb.rdann(
                        str(DATA_FOLDER / "100"),
                        "atr"
                    )

                    normal_symbols = {
                        "N", "L", "R", "e", "j"
                    }

                    abnormal_symbols = {
                        "A", "a", "J", "S",
                        "V", "E", "F", "f"
                    }

                    before = int(0.20 * fs)
                    after = int(0.40 * fs)

                    rows = []

                    for sample, symbol in zip(
                        annotations.sample,
                        annotations.symbol
                    ):

                        if symbol in normal_symbols:
                            reference = "Normal"
                        elif symbol in abnormal_symbols:
                            reference = "Potentially abnormal"
                        else:
                            continue

                        start = sample - before
                        end = sample + after

                        if start < 0 or end > len(ecg):
                            continue

                        beat = ecg[start:end]

                        if len(beat) != before + after:
                            continue

                        prediction = int(
                            model.predict([beat])[0]
                        )

                        prediction_text = (
                            "Potentially abnormal"
                            if prediction == 1
                            else "Normal"
                        )

                        rows.append({
                            "Time (seconds)": round(
                                sample / fs, 3
                            ),
                            "Reference annotation": reference,
                            "Model prediction": prediction_text
                        })

                    if rows:
                        results = pd.DataFrame(rows)

                        st.dataframe(
                            results.head(100),
                            use_container_width=True
                        )

                        st.download_button(
                            "Download AI Classification CSV",
                            data=results.to_csv(index=False),
                            file_name="ecg_classification_results.csv",
                            mime="text/csv"
                        )

                        st.caption(
                            "Reference annotations are dataset labels. "
                            "Model predictions are experimental and "
                            "are not medical diagnoses."
                        )
                    else:
                        st.info(
                            "No eligible annotated beats were found."
                        )

        except Exception as error:
            st.error(f"MIT-BIH analysis failed: {error}")


# =====================================================
# 8. FOOTER
# =====================================================

st.divider()

st.caption(
    "ECG AI Analyzer | Biomedical Engineering Research Prototype"
)
