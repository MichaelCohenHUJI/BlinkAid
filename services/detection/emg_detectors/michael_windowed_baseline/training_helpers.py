import numpy as np
from scipy.stats import mode
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm


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


def train_pca(df, n=3):
    """
    trains a pca model with n components on the supplied dataframe
    :param df: training dataframe
    :param n: number of components
    :return: the data df after pca, the pca model and the scaler object
    """
    # remove the labels
    labels_df = None
    if df.columns[-1] == 'label':
        labels_df = df['label']
        df = df.drop(columns=['label'])
    # init scaler and standardize data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=['timestamp'])), columns=df.columns[1:])
    # put back the timestamp columns
    df_scaled['timestamp'] = df['timestamp']
    cols = ['timestamp'] + [col for col in df_scaled.columns if col != 'timestamp']
    df_scaled = df_scaled[cols]
    # init pca model and train on the data
    pca = PCA(n_components=n)
    df_features = df_scaled.drop(columns=['timestamp'])
    X_pca = pca.fit_transform(df_features)
    pca_columns = [f'PC{i + 1}' for i in range(n)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['timestamp'] = df_scaled['timestamp']
    cols = ['timestamp'] + [col for col in df_pca.columns if col != 'timestamp']
    df_pca = df_pca[cols]
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    # put the label back
    if labels_df is not None:
        df_pca['label'] = labels_df

    return df_pca, pca, scaler


def apply_train_pca(df: pd.DataFrame, scaler: StandardScaler, pca: PCA):
    """
    applies pca transformation to the supplied df
    :param df: data to apply pca on
    :param scaler: standardize scaler model
    :param pca: pca model
    :return: a dataframe with pca transformed df
    """
    pca_columns = [f'PC{i + 1}' for i in range(pca.n_components_)]
    scaled = pd.DataFrame(scaler.transform(df.drop(columns=['timestamp', 'label'])), columns=df.columns[1:-1])
    pcaed = pd.DataFrame(pca.transform(scaled), columns=pca_columns)
    pcaed['timestamp'] = df['timestamp'].values
    pcaed['label'] = df['label'].values
    cols = ['timestamp'] + pca_columns + ['label']
    return pcaed[cols]


def train_ica(df, n=4):
    """
    trains a ica model with n components on the supplied dataframe
    :param df: data to train ica on
    :param n: number of independent components
    :return:
    """
    # remove the labels
    labels_df = None
    if df.columns[-1] == 'label':
        labels_df = df['label']
        df = df.drop(columns=['label'])
    # init and train ica model
    ica = FastICA(n_components=n, random_state=42)
    X_ica = ica.fit_transform(df.drop(columns=['timestamp']))
    df_ica = pd.DataFrame(X_ica, columns=[f"IC{i + 1}" for i in range(n)])
    # add back timestamp and label cols
    df_ica['timestamp'] = df['timestamp']
    cols = ['timestamp'] + [col for col in df_ica.columns if col != 'timestamp']
    df_ica = df_ica[cols]
    if labels_df is not None:
        df_ica['label'] = labels_df

    return df_ica, ica


def apply_train_ica(df: pd.DataFrame, ica: FastICA):
    """
    applies ica transformation to the supplied df
    :param df: the data to apply ica on
    :param ica: trained ica model
    :return: a dataframe with ica transformed df
    """
    n = ica.n_components
    # ICA transform
    icaed = ica.transform(df.drop(columns=['timestamp', 'label']))
    ica_columns = [f'IC{i + 1}' for i in range(n)]
    df_ica = pd.DataFrame(icaed, columns=ica_columns)
    # Add timestamp and label
    df_ica['timestamp'] = df['timestamp'].values
    df_ica['label'] = df['label'].values
    cols = ['timestamp'] + ica_columns + ['label']
    return df_ica[cols]


def create_datasets(train_dfs_processed, val_dfs_processed, window_length, window_overlap, sample_rate):
    """
    creates the train and test datasets design matrices and response vectors
    :param train_dfs_processed: train data after preprocessing
    :param val_dfs_processed: validation data after preprocessing
    :param window_length: window length in seconds
    :param window_overlap: 0-1, overlap fraction between consequent windows
    :param sample_rate: sampling rate in Hz
    :return: train and test datasets design matrices and response vectors
    """
    train_windows = []
    val_windows = []
    for df in tqdm(train_dfs_processed, desc="Windowing train"):
        windows = create_windows(df, window_length, window_overlap, sample_rate)
        train_windows.append(windows)
    for df in tqdm(val_dfs_processed, desc="Windowing test"):
        windows = create_windows(df, window_length, window_overlap, sample_rate)
        val_windows.append(windows)
    train_windows_df = pd.concat(train_windows, ignore_index=True)
    traindf = train_windows_df.sample(frac=1).reset_index(drop=True)
    # Separate features and labels
    X_train = traindf.drop(columns=['timestamp', 'label'])
    y_train = traindf['label']
    if len(val_dfs_processed) > 0:
        val_windows_df = pd.concat(val_windows, ignore_index=True)
        valdf = val_windows_df.sample(frac=1).reset_index(drop=True)
        X_val = valdf.drop(columns=['timestamp', 'label'])
        y_val = valdf['label']
        return X_train, X_val, y_train, y_val
    return X_train, None, y_train, None
