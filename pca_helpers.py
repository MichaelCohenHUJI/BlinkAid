import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from annotationPlots import visualize_channels


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
    pca_columns = [f'PC{i+1}' for i in range(n)]
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
    df_ica = pd.DataFrame(X_ica, columns=[f"IC{i+1}" for i in range(n)])
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


if __name__ == '__main__':
    data_files = {
        'raz': ['2025_03_03_1303_raz_blinks_no_metronome.csv',
                '2025_03_03_1308_raz_left_right.csv',
                '2025_03_03_1311_raz_left_center.csv',
                '2025_03_03_1319_raz_right_center_2.csv',
                '2025_03_03_1322_raz_up_down.csv'],

        'yon': ['annotated_blinks.csv',
                'annotated_eye gaze left right 1.csv']
    }

    data_paths = {'raz': 'data/raz/annotated/annotated_', 'yon': 'data/yonatan/annotated/annotated_',
                  'michael': 'data/michael/'}

    subj = 'raz'

    data_files_paths = [data_paths[subj] + f for f in data_files[subj]]

    df_all = pd.concat((pd.read_csv(f) for f in data_files_paths), ignore_index=True)

    n_comp = 5

    pca_df, _, __ = train_pca(df_all, n_comp)

    for i in range(n_comp):
        dfica, _ = train_ica(df_all, n=i+1)
        visualize_channels(dfica, 'ICA ' + str(i+1))

