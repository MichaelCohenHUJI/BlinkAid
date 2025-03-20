import numpy as np
from scipy.stats import mode
import pandas as pd
import math
from services.detection.emg_detectors.michael_windowed_baseline.labels import LABELS_TO_CLASSES


def create_windows(df, window_length, overlap=0.5, inc_label=True, sample_rate=250):
    """
    Converts sliding windows into a tabular format for ML models like XGBoost or LightGBM.

    Parameters:
    - df (pd.DataFrame): DataFrame where the first column is 'timestamp' and the rest are sensor values.
    - window_length (float): Window length in seconds.
    - overlap (float): Fraction of window overlap (0.0 to 1.0).

    Returns:
    - feature_df (pd.DataFrame): Flattened feature representation of each window.
    """
    window_size = int(window_length * sample_rate)
    step_size = math.ceil(window_size * (1 - overlap))  # Step size for sliding window
    num_samples = df.shape[0]  # Total samples

    timestamps_col = df.columns[0]
    sensor_cols = df.columns[1:-1]  # Exclude timestamp
    label_col = df.columns[-1] if inc_label else None

    timestamps = []  # Store timestamps
    windows_list = []  # Store flattened sensor data
    labels = [] if inc_label else None  # Store labels

    if inc_label and df.columns[-1] != 'label':
        raise ValueError("Dataset must have a 'label' column.")

    # Create list of windows
    for i in range(0, num_samples - window_size, step_size):
        window = df.iloc[i:i + window_size]
        # Assign a label (majority vote)
        window_label = window[label_col].mode().iloc[0] if not window[label_col].empty else None

        # sub-sampling majority class:
        subsamp_ratio = 0.99
        subsamp_prob = 0.75
        if window_label == 0:
            frac = (window[label_col] == window_label).sum() / len(window[label_col])
            if frac < subsamp_ratio or np.random.rand() < subsamp_prob:
                continue

        # Store timestamp of the first row in the window
        timestamp = window[timestamps_col].iloc[0]
        timestamps.append(timestamp)

        # Flatten sensor values
        windows_list.append(window[sensor_cols].values.flatten())

        if inc_label:
            labels.append(window_label)

    # Convert lists to NumPy arrays for efficient DataFrame construction
    timestamps = np.array(timestamps).reshape(-1, 1)
    features_array = np.array(windows_list)  # Shape: (num_windows, num_features)

    df = pd.DataFrame(timestamps, columns=[timestamps_col])

    features_columns = [f"{col}_t{t}" for t in range(window_size) for col in sensor_cols]
    # Construct DataFrame in one step
    dff = pd.DataFrame(features_array, columns=features_columns)

    df = pd.concat([df, dff], axis=1)

    if inc_label:
        labels = np.array(labels).reshape(-1, 1)
        dfl = pd.DataFrame(labels, columns=[label_col])
        df = pd.concat([df, dfl], axis=1)

    return df



