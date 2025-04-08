from itertools import product
from BlinkAidXGB import BlinkAidXGB
from datetime import datetime
from annotated_data_paths import DATA
from pca_helpers import train_pca, apply_train_pca, train_ica, apply_train_ica
from services.common.enums.detection_types import DetectionType
from services.detection.emg_detectors.michael_windowed_baseline import MICHAEL_DETECTOR_DIR
import os
import pandas as pd
from training_helpers import collect_data, create_windows
from tqdm import tqdm


if __name__ == '__main__':
    subj_list = [
        "raz",
        "yon",
        "mich",
        # "noise"
    ]
    trained_on = ''
    for subj in subj_list:
        trained_on += subj + '_'

    split_ratio = 0.2
    p_components = 3
    run_ica = False
    i_components = 3
    sample_rate = 250
    training_window_overlap = 0.99
    inf_window_overlap = 0
    window_length = 0.3
    cooldown = 0.4
    num_channels = 16
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    classes = [DetectionType.NEUTRAL, DetectionType.BLINK, DetectionType.GAZE_LEFT, DetectionType.GAZE_RIGHT,
                         DetectionType.GAZE_CENTER, DetectionType.GAZE_UP, DetectionType.GAZE_DOWN, DetectionType.NOISE]
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

    if run_ica:
        df_all_train_ica, ica = train_ica(df_all_train_pca, n=i_components)
        train_dfs_processed = [apply_train_ica(df, ica) for df in train_dfs_processed]
        val_dfs_processed = [apply_train_ica(df, ica) for df in val_dfs_processed]

    """Stage 3"""
    # create labeled windows from annotated samples
    train_windows = []
    test_windows = []
    for df in tqdm(train_dfs_processed, desc="Windowing train"):
        windows = create_windows(df, window_length, training_window_overlap)
        train_windows.append(windows)
    for df in tqdm(val_dfs_processed, desc="Windowing test"):
        windows = create_windows(df, window_length, training_window_overlap)
        test_windows.append(windows)
    train_windows_df = pd.concat(train_windows, ignore_index=True)
    test_windows_df = pd.concat(test_windows, ignore_index=True)

    """Stage 4"""
    # train model
    # shuffle train & validation sets
    traindf = train_windows_df.sample(frac=1).reset_index(drop=True)
    testdf = test_windows_df.sample(frac=1).reset_index(drop=True)
    # Separate features and labels
    X_train = traindf.drop(columns=['timestamp', 'label'])
    y_train = traindf['label']
    X_test = testdf.drop(columns=['timestamp', 'label'])
    y_test = testdf['label']

    # Define hyperparameter grids
    learning_rates = [0.03, 0.05, 0.1, 0.2]
    max_depths = [6, 7, 8]
    n_estimators = [200, 300, 400, 500]

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
                            sample_rate=sample_rate,
                            training_window_overlap=training_window_overlap,
                            inf_window_overlap=inf_window_overlap,
                            window_length=window_length,
                            cooldown=cooldown,
                            num_channels=num_channels,
                            xgb_params=xgb_params,
                            run_ica=run_ica, i_components=i_components)

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

    # # save model
    # data_frac = str(int((1 - split_ratio) * 100)) + '%data_'
    # model_name = trained_on + data_frac + "xgb_" + str(p_components) + 'pc'
    # model_folder = str(MICHAEL_DETECTOR_DIR) + "/models/" + model_name + "_" + timestamp + "/"
    # os.makedirs(model_folder, exist_ok=True)
    #
    # model_path = model_folder + model_name + "_" + timestamp + ".pkl"
    # best_model.save(model_path)
    #
    # # save models training report
    # with open(model_folder + 'classification_report.txt', 'w') as f:
    #     f.write("Model Accuracy: {:.2f}%\n".format(best_acc * 100))
    #     f.write(f"\nBest Hyperparameters: {best_params}")
    #     f.write("Confusion Matrix:\n")
    #     f.write(str(best_cm) + "\n\n\n")
    #     f.write("Classification Report:\n")
    #     f.write(best_report)
