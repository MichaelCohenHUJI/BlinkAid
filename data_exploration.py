import pandas as pd
from scipy.stats import describe as desc
import matplotlib.pyplot as plt

annotations = {
    'raz': ['raz3-3_lc_ts.csv', 'raz3-3_lr_ts.csv', 'raz3-3_rc2_ts.csv',
            'raz3-3_ud_ts.csv', 'raz_3-3_blinks_ts.csv'],
    'yon': ['ts_blinks_yon23-2.csv', 'ts_eg1_yon23-2.csv']
}


ann_paths = {'raz': 'data/raz_3-3/', 'yon': 'data/yonatan_23-2/'}


ann_files = [ann_paths[subject] + ann for subject in annotations.keys() for ann in annotations[subject]]


means = {}
for file in ann_files:
    df = pd.read_csv(file)
    df = df.loc[~(df['label'] == 'garbage')]
    df['start'] = pd.to_datetime(df['start'])
    df['stop'] = pd.to_datetime(df['stop'])
    df['len'] = df['stop'] - df['start']
    print('in ' + file + ' 0s labels are: ' + str(df.loc[df['len'] >= pd.Timedelta(1, 's')].index))
    df['len'].dt.total_seconds().hist(bins=20)
    plt.title(file)
    plt.show()

    means[file] = desc(df['len'].dt.total_seconds())

print(means)