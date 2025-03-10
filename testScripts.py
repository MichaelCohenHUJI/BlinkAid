# Re-import necessary libraries after execution state reset
import numpy as np
import pandas as pd

# Define parameters
num_samples = 10000  # 10 seconds at 1000 Hz
sampling_rate = 1000  # Hz
window_size = 400  # 0.4 seconds
overlap = 0.5  # 50% overlap
step_size = int(window_size * (1 - overlap))  # Step size for sliding window

# Create timestamps
timestamps = pd.date_range("2025-03-10", periods=num_samples, freq="1ms")

# Generate synthetic EMG sensor data (16 sensors)
sensor_data = np.random.randn(num_samples, 16)
df = pd.DataFrame(sensor_data, columns=[f"sensor_{i + 1}" for i in range(16)])
df.insert(0, "timestamp", timestamps)  # Insert timestamps as the first column

# Simulated event labels (0 = no event, 1 = blink, 2 = look left)
df["event_label"] = np.random.choice([0, 1, 2], size=num_samples)


# Function to create feature windows with timestamps and labels
def create_feature_windows_with_labels(df, window_size, overlap=0.5, label_col="event_label"):
    """
    Converts sliding windows into a tabular format for ML models like XGBoost or LightGBM.
    Keeps the first column (timestamp) and last column (label).

    Parameters:
    - df (pd.DataFrame): DataFrame where the first column is 'timestamp' and the last column is the label.
    - window_size (int): Number of samples per window.
    - overlap (float): Fraction of window overlap (0.0 to 1.0).
    - label_col (str): Name of the column containing event labels.

    Returns:
    - feature_df (pd.DataFrame): Flattened feature representation of each window.
    """
    step_size = int(window_size * (1 - overlap))  # Step size for sliding window
    num_samples = df.shape[0]  # Total samples

    timestamp_col = df.columns[0]  # First column is timestamp
    sensor_cols = df.columns[1:-1]  # Exclude timestamp and label
    feature_list = []
    timestamps = []
    labels = []

    # Create list of windows
    for i in range(0, num_samples - window_size, step_size):
        window = df.iloc[i:i + window_size]

        # Store the timestamp of the first row in the window
        timestamps.append(window[timestamp_col].iloc[0])

        # Flatten sensor values
        features = window[sensor_cols].values.flatten()

        # Assign a label (majority vote)
        label = window[label_col].mode()[0] if not window[label_col].mode().empty else 0

        feature_list.append(np.concatenate([[timestamps[-1]], features, [label]]))  # Add timestamp and label

    # Create column names
    feature_columns = [f"{col}_t{t}" for t in range(window_size) for col in sensor_cols]
    final_columns = [timestamp_col] + feature_columns + [label_col]

    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list, columns=final_columns)

    return feature_df


# Generate feature windows
feature_windows_df = create_feature_windows_with_labels(df, window_size, overlap, label_col="event_label")

