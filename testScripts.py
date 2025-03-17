import os
from pca_ica_exploration import train_pca
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

if __name__ == '__main__':
    ann_data = {
        'raz': ['annotated_2025_03_03_1303_raz_blinks_no_metronome.csv',
                'annotated_2025_03_03_1308_raz_left_right.csv',
                'annotated_2025_03_03_1311_raz_left_center.csv',
                'annotated_2025_03_03_1319_raz_right_center_2.csv',
                'annotated_2025_03_03_1322_raz_up_down.csv'],

        'yon': ['annotated_blinks.csv',
                'annotated_eye gaze left right 1.csv']
    }
    folder_paths = {'raz': 'data/raz_3-3/annotated/', 'yon': 'data/yonatan_23-2/annotated/'}
    subj = 'raz'
    model_name = "raz_xg_windowed_stdized_16pc"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_meta = {}

    # collect data from all files
    ann_data_paths = [folder_paths[subj] + f for f in ann_data[subj]]
    df = pd.concat((pd.read_csv(f) for f in ann_data_paths), ignore_index=True)

    # run pca on the concatenated data
    p_components = 3
    model_meta['p_components'] = p_components
    df, pca_results, pca, scaler = train_pca(df, p_components)

    model_name = "raz_xg_windowed_stdized_" + str(p_components) + 'pc'
    model_folder = "models/" + model_name + "_" + timestamp + "/"
    os.makedirs(model_folder, exist_ok=True)

    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(model_folder, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Log PCA explained variance to TensorBoard
    for i, var in enumerate(pca.explained_variance_ratio_):
        writer.add_scalar('PCA/Explained_Variance_Ratio_PC' + str(i+1), var, 0)

    # save pca and scaler data
    scaler_path = model_folder + model_name + "_" + timestamp + "_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    model_meta['scaler_path'] = scaler_path
    pca_model_path = model_folder + model_name + "_" + timestamp + "_pca_model.pkl"
    joblib.dump(pca, pca_model_path)
    model_meta['pca_model_path'] = pca_model_path

    # create labeled windows from annotated samples
    window_length = 0.3
    overlap = 0.7
    model_meta['window_length'] = window_length
    model_meta['overlap'] = overlap
    windows = create_windows(df, window_length, overlap)

    # train model
    existing_model = 0
    n_classes = 7
    model_meta['n_classes'] = n_classes
    trained_model, cm, report, report_dict = train_xgb(windows, n_classes)

    # Log classification metrics to TensorBoard
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            writer.add_scalar(f'Classification_Report/{label}_precision', metrics['precision'], 0)
            writer.add_scalar(f'Classification_Report/{label}_recall', metrics['recall'], 0)
            writer.add_scalar(f'Classification_Report/{label}_f1-score', metrics['f1-score'], 0)
        else:
            writer.add_scalar('Classification_Report/accuracy', report_dict['accuracy'], 0)

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

    cm_image = plot_confusion_matrix(cm, labels=[str(i) for i in range(n_classes)])
    writer.add_image('Confusion_Matrix', cm_image[0], 0)

    # save model
    model_path = model_folder + model_name + "_" + timestamp + ".pkl"
    joblib.dump(trained_model, model_path)
    print(f"Trained model saved to {model_path}")

    meta_path = model_folder + model_name + "_" + timestamp + "_metadata.pkl"
    joblib.dump(model_meta, meta_path)

    # Close TensorBoard writer
    writer.close()

    with open(model_folder + 'classification_report.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n\n")
        f.write("Classification Report:\n")
        f.write(report)