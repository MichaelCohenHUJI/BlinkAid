import numpy as np
from scipy.stats import mode
import pandas as pd
import math


def collect_data(data_paths_dict, subj_list, split_ratio):
    """
    creates a list of dataframes containing the training data from the files in the provided paths
    :param data_paths_dict: data paths dictionary, keys are subject names and values are the paths to the data files
    :param subj_list: the list of subject names on which to train the data
    :param split_ratio: 0 - 1 float, the fraction of the data to be used for validation
    :return: list of training data dataframes and validation data dataframes
    """
    train_dfs = []
    test_dfs = []
    for subj in data_paths_dict:
        # if current subject is not in the list provided do not use its data
        if subj not in subj_list:
            continue
        for file_path in data_paths_dict[subj]:
            df = pd.read_csv(file_path)
            split_idx = int(len(df) * (1 - split_ratio))
            train_dfs.append(df.iloc[:split_idx])
            test_dfs.append(df.iloc[split_idx:])
    return train_dfs, test_dfs


def create_windows(df, window_length, overlap=0.5, sample_rate=250):
    """
    Creates windows from a dataframe.
    :param df: DataFrame where the first column is 'timestamp' followed by sensor values and then the label.
    :param window_length: Window length in seconds.
    :param overlap: Fraction of window overlap (0.0 to 1.0).
    :param sample_rate: default - 250hz
    :return: Flattened feature representation of each window. first column is the timestamp, last is the label.
    """
    window_size = int(window_length * sample_rate)  # size in samples number
    step_size = math.ceil(window_size * (1 - overlap))  # Step size for sliding window
    num_samples = df.shape[0]  # Total samples

    timestamps_col = df.columns[0]
    sensor_cols = df.columns[1:-1]  # Exclude timestamp and label
    label_col = df.columns[-1]

    timestamps = []  # Store timestamps
    windows_list = []  # Store flattened sensor data
    labels = []  # Store labels

    if df.columns[0] != 'timestamp':
        raise ValueError("Dataset must have a 'timestamp' column.")
    if df.columns[-1] != 'label':
        raise ValueError("Dataset must have a 'label' column.")

    # Create list of windows
    for i in range(0, num_samples - window_size, step_size):
        window = df.iloc[i:i + window_size]

        # Assign a label (majority vote)
        window_label = window[label_col].mode().iloc[0] if not window[label_col].empty else None

        # sub-sampling majority class:
        subsamp_ratio = 0.99
        subsamp_prob = 0.75
        if window_label == 0:  # neutral
            # fraction of samples in the window that are classified as neutral
            frac = (window[label_col] == window_label).sum() / len(window[label_col])
            if frac < subsamp_ratio or np.random.rand() < subsamp_prob:
                # if fraction of neutral label of the window is less than sub-sampling ratio - discard it
                # if fraction is => sub-sampling ratio keep it with probability of 25%
                continue

        # Store timestamp of the first row in the window
        timestamp = window[timestamps_col].iloc[0]
        timestamps.append(timestamp)

        # Flatten sensor values
        windows_list.append(window[sensor_cols].values.flatten())
        # save window label
        labels.append(window_label)

    # Convert lists to NumPy arrays for efficient DataFrame construction
    timestamps = np.array(timestamps).reshape(-1, 1)
    df = pd.DataFrame(timestamps, columns=[timestamps_col])
    # create window columns
    features_columns = [f"{col}_t{t}" for t in range(window_size) for col in sensor_cols]
    # Construct DataFrame in one step
    features_array = np.array(windows_list)
    dff = pd.DataFrame(features_array, columns=features_columns)
    # combine both dfs
    df = pd.concat([df, dff], axis=1)
    # add the labels
    labels = np.array(labels).reshape(-1, 1)
    dfl = pd.DataFrame(labels, columns=[label_col])
    df = pd.concat([df, dfl], axis=1)

    return df

