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
annotations = {
    'raz': ['raz3-3_lc_ts.csv', 'raz3-3_lr_ts.csv', 'raz3-3_rc2_ts.csv',
            'raz3-3_ud_ts.csv', 'raz_3-3_blinks_ts.csv'],
    'yon': ['ts_blinks_yon23-2.csv', 'ts_eg1_yon23-2.csv']
}


ann_paths = {'raz': 'data/raz_3-3/', 'yon': 'data/yonatan_23-2/'}


ann_files = [ann_paths[subject] + ann for subject in annotations.keys() for ann in annotations[subject]]

#%%
yon = 'data/yonatan_23-2'
raz = 'data/raz_3-3'
michael = 'data/michael_3-3'
filename = '2025_03_03_1340_raz_wink_left_right.csv'
filepath = os.path.join(raz, filename)
df = pd.read_csv(filepath)
#%%
# plot data
visualize_channels(df, filename + ' original data')


#%%
# run PCA
file_names = []
for f in os.listdir(raz):
    if f.endswith('.csv') and f.startswith('2025'):
        file_names.append(f)

#%%
# standardized data
def run_pca(df, n=8):
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
    
    return df_pca, pca_result
    







#%%
# for f in file_names:
#     cur_df = pd.read_csv(os.path.join(raz, f))
#     print(f + f' ')
#     run_pca(df, 8)
#     print()
    
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
        df_pca, X = run_pca(df,n)
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
    for n in range(5):
        ica, _ = run_ica(df, n+1, False)
        visualize_channels(ica, filename + ' ICA ' + str(n))

    # df_pca, X = run_pca(df, 8)

    # visualize_channels(df_pca, filename + ' PCA')
    # plot_ica(ica, 5)


