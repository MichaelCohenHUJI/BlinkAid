import numpy as np
from scipy.stats import mode
import pandas as pd


def create_windows(df, window_length, overlap=0.5, inc_label=True):
    """
    Converts sliding windows into a tabular format for ML models like XGBoost or LightGBM.

    Parameters:
    - df (pd.DataFrame): DataFrame where the first column is 'timestamp' and the rest are sensor values.
    - window_length (float): window length in seconds.
    - overlap (float): Fraction of window overlap (0.0 to 1.0).

    Returns:
    - feature_df (pd.DataFrame): Flattened feature representation of each window.
    """
    window_size = int(window_length * 250)
    step_size = int(window_size * (1 - overlap))  # Step size for sliding window
    num_samples = df.shape[0]  # Total samples

    timestamps_col = df.columns[0]

    sensor_cols = df.columns[1:-1]  # Exclude timestamp
    feature_list = []
    windows_timestamps = []
    labels = []
    feature_columns = [f"{col}_t{t}" for t in range(window_size) for col in sensor_cols]
    final_columns = [timestamps_col] + feature_columns
    if inc_label:
        if df.columns[-1] != 'label':
            raise ValueError("Dataset must have a 'label' column.")
        label_col = df.columns[-1]
        final_columns = final_columns + [label_col]

    # Create list of windows
    for i in range(0, num_samples - window_size, step_size):
        window = df.iloc[i:i + window_size]

        # Store the timestamp of the first row in the window
        windows_timestamps.append(window[timestamps_col].iloc[0])

        # Flatten sensor values
        features = window[sensor_cols].values.flatten()

        if inc_label:
            # Assign a label (majority vote)
            window_label = mode(window[label_col].values, axis=None, keepdims=True).mode[0] if not window[label_col].empty else 0
            feature_list.append(np.concatenate([[windows_timestamps[-1]], features, [window_label]]))  # Add timestamp and label

        else:
            feature_list.append(np.concatenate([[windows_timestamps[-1]], features]))


    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list, columns=final_columns)

    return feature_df


