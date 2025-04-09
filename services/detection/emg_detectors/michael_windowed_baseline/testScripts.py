from itertools import product
from services.detection.emg_detectors.michael_windowed_baseline.BlinkAidXGB import BlinkAidXGB
from datetime import datetime
from annotated_data_paths import DATA
from services.common.enums.detection_types import DetectionType
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
import os
import pandas as pd
from training_helpers import collect_data, create_windows, train_pca, apply_train_pca, create_datasets
from tqdm import tqdm

if __name__ == '__main__':
    subj_list = [
        "raz",
        "yon",
        "mich",
        "noise"
    ]
    trained_on = ''
    for subj in subj_list:
        trained_on += subj + '_'

    split_ratio = 0.2
    p_components = 3
    training_window_overlap = 0.99
    inf_window_overlap = 0
    window_length = 0.3
    cooldown = 0.6
    sample_rate = 250

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    classes = [DetectionType.NEUTRAL, DetectionType.BLINK, DetectionType.GAZE_LEFT, DetectionType.GAZE_RIGHT,
               # DetectionType.GAZE_CENTER,
               DetectionType.GAZE_UP, DetectionType.GAZE_DOWN, DetectionType.NOISE]
    data_paths = DATA

    """Stage 1"""
    # collect data from all files
    train_dfs, val_dfs = collect_data(data_paths, subj_list, split_ratio)

    """Stage 2"""
    # train standardization and pca/ica models on the train data
    df_all_train = pd.concat(train_dfs, ignore_index=True)

    df_all_train_pca, pca, scaler = train_pca(df_all_train, p_components)

    # apply pca to whole data
    train_dfs_processed = [apply_train_pca(df, scaler, pca) for df in train_dfs]
    val_dfs_processed = [apply_train_pca(df, scaler, pca) for df in val_dfs]

    """Stage 3"""
    # create labeled windows from annotated samples
    X_train, X_test, y_train, y_test = create_datasets(train_dfs_processed, val_dfs_processed, window_length,
                                                       training_window_overlap, sample_rate)

    # Define hyperparameter grids
    learning_rates = [0.03, 0.05, 0.1, 0.15]
    max_depths = [5, 6, 7, 8]
    n_estimators = [100, 200, 300, 400, 500]

    # Before the loop
    best_score = 0  # Best F1 Score now
    best_params = None
    best_model = None
    best_report = None
    best_report_dict = None
    best_acc = None
    best_cm = None

    # Generate all combinations
    param_grid = list(product(learning_rates, max_depths, n_estimators))

    for lr, depth, n_est in param_grid:
        print(f"\nTraining with lr={lr}, max_depth={depth}, n_estimators={n_est}")

        # Define xgb_params
        xgb_params = {
            'learning_rate': lr,
            'max_depth': depth,
            'n_estimators': n_est
        }

        # Create new model
        model = BlinkAidXGB(classes,
                            split_ratio=split_ratio,
                            p_components=p_components,
                            training_window_overlap=training_window_overlap,
                            inf_window_overlap=inf_window_overlap,
                            window_length=window_length,
                            cooldown=cooldown,
                            xgb_params=xgb_params)

        # Train
        model.fit_hpt(X_train, y_train, X_test, y_test)

        # Get validation predictions manually
        acc, cm, report, report_dict = model.get_performance_report()

        # Extract y_true and y_pred manually (optional way)
        # OR parse from report_dict:
        macro_f1 = report_dict['macro avg']['f1-score']  # <-- THIS IS THE MACRO F1

        print(f"Macro F1 Score: {macro_f1:.4f}")

        # Select best based on macro F1
        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = xgb_params
            best_model = model
            best_report_dict = report_dict
            best_acc = acc

    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best Validation Macro F1 Score: {best_score:.4f}")

    # save model
    data_frac = str(int((1 - split_ratio) * 100)) + '%data_'
    model_name = trained_on + data_frac + "xgb_" + str(p_components) + 'pc'
    model_folder = str(MICHAEL_DETECTOR_DIR) + "/hyper_parameter_results/"

    model_path = model_folder + model_name + "_" + timestamp + ".pkl"

    # save models training report
    with open(model_folder + 'hpt_report.txt', 'w') as f:
        f.write("Model Accuracy: {:.2f}%\n".format(best_acc * 100))
        f.write(f"\nBest Hyperparameters: {best_params}")
        f.write("Confusion Matrix:\n")
        f.write(str(best_cm) + "\n\n\n")
        f.write("Classification Report:\n")
        f.write(best_report)
