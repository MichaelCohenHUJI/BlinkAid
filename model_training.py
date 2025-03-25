import os
from pca_ica_exploration import train_pca
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
from windowing import create_windows
import pandas as pd
from firstModel import train_xgb
from datetime import datetime
import joblib
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import io
import torch
from tqdm import tqdm
from annotated_data_paths import ann_data, folder_paths


if __name__ == '__main__':
    """
    main framework for training. 
    Stages:
    1. collect data (train + test)
    2. trains standardization and pca models on train set and applies them on all data
    3. creates sliding windows for training and test sets
    4. training an xgboost model on train set, and returning a performance report on test set
    5. saving the models mentioned above, the performance report and creates a tensorboard report for the trained model
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    """Stage 1"""
    # collect data from all files
    train_dfs = []
    test_dfs = []
    split_ratio = 0.2
    trained_on = ''
    for subj in ann_data:
        # if subj != 'raz':
        #     continue
        trained_on += subj + '_'
        ann_data_paths = [folder_paths[subj] + f for f in ann_data[subj]]
        for file_path in ann_data_paths:
            df = pd.read_csv(file_path)
            split_idx = int(len(df) * (1 - split_ratio))
            train_dfs.append(df.iloc[:split_idx])
            test_dfs.append(df.iloc[split_idx:])
    # train standardization and pca models on the train data
    df_all = pd.concat(train_dfs, ignore_index=True)
    p_components = 3
    df_all_pca, pca_results, pca, scaler = train_pca(df_all, p_components)

    """Stage 2"""
    # apply pca to whole data
    pca_cols = [f'PC{i+1}' for i in range(p_components)]
    cols = ['timestamp'] + pca_cols + ['label']
    def apply_pca(df):
        scaled = pd.DataFrame(scaler.transform(df.drop(columns=['timestamp', 'label'])), columns=df.columns[1:-1])
        pcaed = pd.DataFrame(pca.transform(scaled), columns=pca_cols)
        pcaed['timestamp'] = df['timestamp'].values
        pcaed['label'] = df['label'].values
        return pcaed[cols]
    train_dfs_pca = [apply_pca(df) for df in train_dfs]
    test_dfs_pca = [apply_pca(df) for df in test_dfs]

    """Stage 3"""
    # create labeled windows from annotated samples
    window_length = 0.3  # seconds
    overlap = 0.99  # 0 - 1
    train_windows = []
    test_windows = []
    for df in tqdm(train_dfs_pca):
        windows = create_windows(df, window_length, overlap)
        train_windows.append(windows)
    for df in tqdm(test_dfs_pca):
        windows = create_windows(df, window_length, overlap)
        test_windows.append(windows)
    train_windows_df = pd.concat(train_windows, ignore_index=True)
    test_windows_df = pd.concat(test_windows, ignore_index=True)

    """Stage 4"""
    # train model
    existing_model = 0
    n_classes = 7
    trained_model, cm, report, report_dict = train_xgb(train_windows_df, test_windows_df, n_classes)

    """Stage 5"""
    # create model folder
    data_frac = str(int(split_ratio * 100)) + '%data_'
    model_name = trained_on + data_frac + "xg_windowed_stdized_" + str(p_components) + 'pc'
    model_folder = str(MICHAEL_DETECTOR_DIR) + "/models/" + model_name + "_" + timestamp + "/"
    os.makedirs(model_folder, exist_ok=True)

    # save models training report
    with open(model_folder + 'classification_report.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # save pca and scaler data
    scaler_path = model_folder + model_name + "_" + timestamp + "_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    pca_model_path = model_folder + model_name + "_" + timestamp + "_pca_model.pkl"
    joblib.dump(pca, pca_model_path)

    # save model metadata
    model_meta = {}
    model_meta['scaler_path'] = scaler_path
    model_meta['p_components'] = p_components
    model_meta['pca_model_path'] = pca_model_path
    model_meta['window_length'] = window_length
    model_meta['overlap'] = overlap
    model_meta['n_classes'] = n_classes

    # save model
    model_path = model_folder + model_name + "_" + timestamp + ".pkl"
    joblib.dump(trained_model, model_path)
    print(f"Trained model saved to {model_path}")

    meta_path = model_folder + model_name + "_" + timestamp + "_metadata.pkl"
    joblib.dump(model_meta, meta_path)


    # 🔥 Add Confusion Matrix Heatmap
    def plot_confusion_matrix(cm, labels):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        image = torch.tensor(plt.imread(buf)).permute(2, 0, 1)[:3]  # [C, H, W]
        return image.unsqueeze(0)  # [1, C, H, W]


    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(model_folder, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Log PCA explained variance to TensorBoard
    for i, var in enumerate(pca.explained_variance_ratio_):
        writer.add_scalar('PCA/Explained_Variance_Ratio_PC' + str(i + 1), var, 0)

    # Log classification metrics to TensorBoard
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            writer.add_scalar(f'Classification_Report/{label}_precision', metrics['precision'], 0)
            writer.add_scalar(f'Classification_Report/{label}_recall', metrics['recall'], 0)
            writer.add_scalar(f'Classification_Report/{label}_f1-score', metrics['f1-score'], 0)
        else:
            writer.add_scalar('Classification_Report/accuracy', report_dict['accuracy'], 0)

    cm_image = plot_confusion_matrix(cm, labels=[str(i) for i in range(n_classes)])
    writer.add_image('Confusion_Matrix', cm_image[0], 0)

    # Close TensorBoard writer
    writer.close()

