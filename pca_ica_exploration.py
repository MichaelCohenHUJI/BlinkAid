#%%
import pandas as pd
from scipy.stats import describe as desc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
import os
from firstPlots import visualize_channels

#%%

def train_pca(df, n=8):
    labels_df = None
    if df.columns[-1] == 'label':
        labels_df = df['label']
        df = df.drop(columns=['label'])

    scaler = preprocessing.StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=['timestamp'])), columns=df.columns[1:])
    
    df_scaled['timestamp'] = df['timestamp']
    cols = ['timestamp'] + [col for col in df_scaled.columns if col != 'timestamp']
    df_scaled = df_scaled[cols]
    
    pca = PCA(n_components=n)
    df_features = df_scaled.drop(columns=['timestamp'])
    pca_result = pca.fit_transform(df_features)
    pca_columns = [f'PC{i+1}' for i in range(n)]
    df_pca = pd.DataFrame(pca_result, columns=pca_columns)
    df_pca['timestamp'] = df_scaled['timestamp']
    cols = ['timestamp'] + [col for col in df_pca.columns if col != 'timestamp']
    df_pca = df_pca[cols]
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))

    if labels_df is not None:
        df_pca['label'] = labels_df
    
    return df_pca, pca_result, pca, scaler
    




#%%
def run_ica(df, n=8, do_pca=True):
    """
    Run ICA on X, which is the data after pca
    :param X: 
    :param n: number of ica components
    :return: 
    """
    X = None
    if do_pca:
        df_pca, X, _, __ = train_pca(df, n)
    else:
        scaler = preprocessing.StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=['timestamp'])), columns=df.columns[1:])

        df_scaled['timestamp'] = df['timestamp']
        cols = ['timestamp'] + [col for col in df_scaled.columns if col != 'timestamp']
        df_scaled = df_scaled[cols]
        X = df_scaled.drop(columns=['timestamp'])

    ica = FastICA(n_components=n, random_state=42)
    X_ica = ica.fit_transform(X)
    df_ica = pd.DataFrame(X_ica, columns=[f"IC{i+1}" for i in range(n)])
    df_ica['timestamp'] = df['timestamp']
    cols = ['timestamp'] + [col for col in df_ica.columns if col != 'timestamp']
    df_ica = df_ica[cols]
    
    return df_ica, X_ica



#%%
def plot_ica(df, n_components=8):

    # Define grid layout (3x3 but using 8 subplots)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()  # Flatten to loop easily

    # Define grid layout (3x3 but using 8 subplots)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()  # Flatten to loop easily

    # Plot each Independent Component separately
    for i in range(n_components):
        axes[i].plot(df['timestamp'], df[f"IC{i + 1}"], color='b')
        axes[i].set_title(f'Independent Component {i + 1}')
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amplitude")
        axes[i].tick_params(axis='x', rotation=45)

    # Remove extra subplot if any
    for j in range(n_components, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()
    

#%%
if __name__ == '__main__':
    # %%
    data_files = {
        'raz': ['2025_03_03_1303_raz_blinks_no_metronome.csv',
                '2025_03_03_1308_raz_left_right.csv',
                '2025_03_03_1311_raz_left_center.csv',
                '2025_03_03_1319_raz_right_center_2.csv',
                '2025_03_03_1322_raz_up_down.csv'],

        'yon': ['annotated_blinks.csv',
                'annotated_eye gaze left right 1.csv']
    }

    data_paths = {'raz': 'data/raz_3-3/annotated/annotated_', 'yon': 'data/yonatan_23-2/annotated/annotated_', 'michael': 'data/michael_3-3/'}

    subj = 'raz'

    data_files_paths = [data_paths[subj] + f for f in data_files[subj]]

    # %%
    df = pd.concat((pd.read_csv(f) for f in data_files_paths), ignore_index=True)
    labels_df = None
    if df.columns[-1] == 'label':
        labels_df = df['label']
        df = df.drop(columns=['label'])

    # %%
    # plot data
    visualize_channels(df, subj + ' original data')
    pca_df, X, pca, scaler = train_pca(df, 3)
    visualize_channels(pca_df, subj + ' pca')



    # %%
    # for f in file_names:
    #     cur_df = pd.read_csv(os.path.join(raz, f))
    #     print(f + f' ')
    #     run_pca(df, 8)
    #     print()

    # %%
    # run PCA
    # file_names = []
    # for f in os.listdir(raz):
    #     if f.endswith('.csv') and f.startswith('2025'):
    #         file_names.append(f)
    # for n in range(5):
    #     ica, _ = run_ica(df, n+1, False)
    #     visualize_channels(ica, filename + ' ICA ' + str(n))

    # df_pca, X = run_pca(df, 8)

    # visualize_channels(df_pca, filename + ' PCA')
    # plot_ica(ica, 5)


