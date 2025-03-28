import pandas as pd


def collect_data(data_paths_dict, subj_list, split_ratio):
    train_dfs = []
    test_dfs = []
    for subj in data_paths_dict:
        if subj not in subj_list:
            continue
        ann_data_paths = [p for p in data_paths_dict[subj]]
        for file_path in ann_data_paths:
            df = pd.read_csv(file_path)
            split_idx = int(len(df) * (1 - split_ratio))
            train_dfs.append(df.iloc[:split_idx])
            test_dfs.append(df.iloc[split_idx:])
    return train_dfs, test_dfs
